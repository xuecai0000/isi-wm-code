from time import perf_counter
import json

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer
from common import object_state_supervision
from envs.wrappers.object_state_supervision import (
	find_state_supervision_wrapper, without_state_labels,
)


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._eval_idx = 0
		# perf_counter is monotonic and therefore suitable for reporting actual
		# wall-clock runtime even if the host clock is adjusted during a run.
		self._start_time = perf_counter()
		self._eval_time = 0.

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = perf_counter() - self._start_time
		train_elapsed_time = max(0., elapsed_time - self._eval_time)
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			wallclock_elapsed_time=elapsed_time,
			train_elapsed_time=train_elapsed_time,
			total_eval_elapsed_time=self._eval_time,
			steps_per_second=self._step / max(elapsed_time, 1e-12),
			train_steps_per_second=self._step / max(train_elapsed_time, 1e-12),
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		with without_state_labels(self.env):
			return self._eval_without_state_labels()

	def _eval_without_state_labels(self):
		"""Evaluation rollout while privileged label collection is suspended."""
		eval_start = perf_counter()
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			reset_belief = getattr(self.agent, 'reset_object_belief', None)
			if callable(reset_belief):
				reset_belief()
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		eval_elapsed_time = perf_counter() - eval_start
		self._eval_time += eval_elapsed_time
		if bool(self.cfg.get('save_eval_episode_trace', False)):
			rewards = [float(value) for value in ep_rewards]
			successes = [float(value) for value in ep_successes]
			lengths = [int(value) for value in ep_lengths]
			payload = {
				'evaluation_index': self._eval_idx,
				'step': self._step,
				'episodes': len(rewards),
				'episode_rewards': rewards,
				'episode_successes': successes,
				'episode_lengths': lengths,
				'reward_mean': float(np.nanmean(rewards)),
				'reward_std': float(np.nanstd(rewards)),
				'reward_min': float(np.nanmin(rewards)),
				'reward_max': float(np.nanmax(rewards)),
			}
			trace_path = self.cfg.work_dir / 'eval_episodes.jsonl'
			with trace_path.open('a', encoding='utf-8', newline='\n') as stream:
				stream.write(json.dumps(payload, allow_nan=False) + '\n')
			self._eval_idx += 1
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
			eval_elapsed_time=eval_elapsed_time,
		)

	def _anchor_runtime_metrics(self):
		"""Return locator runtime counters from an optional anchor wrapper."""
		if (
			not self.cfg.get('flat_anchor', False)
			and not self.cfg.get('cutie_masked_rgb_enabled', False)
			and not self.cfg.get('cutie_mask_guided_rgb_enabled', False)
		):
			return None
		env, visited = self.env, set()
		while env is not None and id(env) not in visited:
			visited.add(id(env))
			metrics = getattr(env, 'metrics', None)
			if callable(metrics):
				try:
					return metrics()
				except Exception as exc:
					return {'error': repr(exc)}
			teacher = getattr(env, 'teacher', None)
			metrics = getattr(teacher, 'metrics', None)
			if callable(metrics):
				try:
					return metrics()
				except Exception as exc:
					return {'error': repr(exc)}
			env = getattr(env, 'env', None)
		return None

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		if object_state_supervision.enabled(self.cfg):
			wrapper = find_state_supervision_wrapper(self.env)
			if wrapper is None:
				raise RuntimeError('State-supervised replay requires a label wrapper.')
			target = torch.from_numpy(wrapper.get_object_state_target())
			td[object_state_supervision.REPLAY_KEY] = target.unsqueeze(0)
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0 or self._step == self.cfg.steps:
				eval_next = True

			# Reset environment
			if done:
				final_eval_complete = False
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					if (
						self._step > 0
						and bool(self.cfg.get('save_eval_checkpoints', False))
					):
						self.logger.save_agent(
							self.agent, identifier=f'eval_{self._step}'
						)
					eval_next = False
					final_eval_complete = self._step >= self.cfg.steps

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				# Preserve logging/replay insertion for the episode that ended at the
				# final boundary, then stop before reset, planning, or another update.
				if final_eval_complete:
					break

				obs = self.env.reset()
				reset_belief = getattr(self.agent, 'reset_object_belief', None)
				if callable(reset_belief):
					reset_belief()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		state_wrapper = find_state_supervision_wrapper(self.env)
		if state_wrapper is not None:
			with (self.cfg.work_dir / 'object_state_supervision_runtime.json').open(
				'w', encoding='utf-8'
			) as stream:
				json.dump(state_wrapper.state_supervision_metrics(), stream, indent=2, allow_nan=False)
				stream.write('\n')
		anchor_metrics = self._anchor_runtime_metrics()
		if anchor_metrics is not None:
			print('ANCHOR_RUNTIME', anchor_metrics)
			with open(
				self.cfg.work_dir / 'perception_runtime.json',
				'w', encoding='utf-8',
			) as file:
				json.dump(
					anchor_metrics, file, ensure_ascii=False, indent=2,
					default=str, allow_nan=False,
				)
				file.write('\n')
		replay_metrics = getattr(self.buffer, 'metrics', None)
		if isinstance(replay_metrics, dict):
			with open(
				self.cfg.work_dir / 'replay_runtime.json',
				'w', encoding='utf-8',
			) as file:
				json.dump(
					replay_metrics, file, ensure_ascii=False, indent=2,
					default=str, allow_nan=False,
				)
				file.write('\n')
		# Capture pure train/eval completion before final checkpoint serialization.
		# This lets structural speed pilots compare learning throughput without
		# conflating it with backend-dependent artifact size and filesystem speed.
		training_elapsed = perf_counter() - self._start_time
		training_non_eval = max(0., training_elapsed - self._eval_time)
		self.logger.finish(self.agent)
		wallclock = perf_counter() - self._start_time
		wallclock_metrics = {
			'seed': int(self.cfg.seed),
			'elapsed_seconds': round(wallclock, 3),
			'eval_seconds': round(self._eval_time, 3),
			'non_eval_seconds': round(max(0., wallclock - self._eval_time), 3),
			'training_elapsed_seconds': round(training_elapsed, 3),
			'training_non_eval_seconds': round(training_non_eval, 3),
			'checkpoint_finalize_seconds': round(
				max(0., wallclock - training_elapsed), 3
			),
			'steps': int(self._step),
			'non_eval_steps_per_second': float(
				self._step / max(wallclock - self._eval_time, 1e-12)
			),
			'training_non_eval_steps_per_second': float(
				self._step / max(training_non_eval, 1e-12)
			),
		}
		with open(
			self.cfg.work_dir / 'trainer_runtime.json',
			'w', encoding='utf-8',
		) as file:
			json.dump(
				wallclock_metrics, file, ensure_ascii=False, indent=2,
				default=str, allow_nan=False,
			)
			file.write('\n')
		print('TRAINER_WALLCLOCK', wallclock_metrics)

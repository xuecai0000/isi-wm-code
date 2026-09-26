import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from common import object_state_supervision
from common import mask_guided_rgb
from common import robust_object_field


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._state_supervision_enabled = object_state_supervision.enabled(cfg)
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._belief_sampler = None
		self._belief_replay_generator = None
		self._belief_enabled = bool(
			cfg.get('flat_anchor', False)
			and cfg.get('flat_anchor_mode', 'residual') == 'cutie_object_only'
			and cfg.get('cutie_object_belief_enabled', False)
		)
		if self._belief_enabled:
			belief_batch_size = int(cfg.get(
				'cutie_object_belief_batch_size', cfg.batch_size
			))
			if belief_batch_size < 2:
				raise ValueError(
					'Cutie belief batch size must be at least two, got '
					f'{belief_batch_size}.'
				)
			self._belief_sequence_length = (
				int(cfg.get('cutie_object_belief_burn_in', 3))
				+ int(cfg.get('cutie_object_belief_max_burst', 20))
				+ int(cfg.get('cutie_object_belief_recovery_frames', 1))
			)
			if self._belief_sequence_length < 3:
				raise ValueError('Cutie belief replay sequence is too short.')
			self._belief_batch_size = (
				belief_batch_size * self._belief_sequence_length
			)
			# Long belief slices must not advance the ordinary TD-MPC replay
			# sampler/global RNG. Otherwise enabling the auxiliary changes every
			# subsequent base training batch and confounds the ablation.
			self._belief_sampler = SliceSampler(
				num_slices=belief_batch_size,
				end_key=None,
				traj_key='episode',
				truncated_key=None,
				strict_length=True,
				cache_values=False,
			)
		else:
			self._belief_sequence_length = None
			self._belief_batch_size = None
		self._num_eps = 0
		self._bytes_per_step = None
		self._storage_required_bytes = None
		self._storage_device = None
		self._observation_keys = None

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	@property
	def metrics(self):
		"""Return replay placement evidence for runtime/speed comparisons."""
		return {
			'capacity': int(self._capacity),
			'bytes_per_step': (
				float(self._bytes_per_step)
				if self._bytes_per_step is not None else None
			),
			'storage_required_bytes': (
				int(self._storage_required_bytes)
				if self._storage_required_bytes is not None else None
			),
			'storage_device': (
				str(self._storage_device)
				if self._storage_device is not None else None
			),
			'observation_keys': self._observation_keys,
			'object_state_target_dim': (
				object_state_supervision.target_dim(self.cfg)
				if self._state_supervision_enabled else None
			),
			'object_state_target_in_policy_observation': False,
			'belief_sequence_length': self._belief_sequence_length,
			'belief_batch_size': (
				int(self.cfg.get('cutie_object_belief_batch_size', self.cfg.batch_size))
				if self._belief_enabled else None
			),
			'belief_replay_rng_isolated': bool(
				self._belief_enabled
				and self._belief_sampler is not None
				and self._belief_replay_generator is not None
			),
		}

	def _validate_observation_schema(self, obs, context):
		"""Fail closed on the selected structural observation boundary."""
		if mask_guided_rgb.enabled(self.cfg):
			if not isinstance(obs, TensorDict):
				raise TypeError(
					f'Mask-guided RGB {context} observation must be a TensorDict.'
				)
			expected = mask_guided_rgb.observation_shapes(self.cfg)
			keys = set(obs.keys())
			if keys != set(expected):
				raise ValueError(
					f'Mask-guided RGB {context} keys {sorted(keys)} do not '
					f'match {sorted(expected)}.'
				)
			for key, shape in expected.items():
				if tuple(obs[key].shape[-len(shape):]) != shape:
					raise ValueError(
						f'Mask-guided RGB {context} {key} shape must end in '
						f'{shape}, got {tuple(obs[key].shape)}.'
					)
			if obs['rgb'].dtype != torch.uint8:
				raise ValueError(f'Mask-guided RGB {context} RGB must be uint8.')
			if obs['object_mask'].dtype != torch.bool:
				raise ValueError(
					f'Mask-guided RGB {context} masks must be boolean.'
				)
			status = obs['tracker_status']
			if status.dtype != torch.float32:
				raise ValueError(
					f'Mask-guided RGB {context} tracker status must be float32.'
				)
			if context == 'replay insertion' and not bool(
				torch.isfinite(status).all().item()
			):
				raise ValueError(
					f'Mask-guided RGB {context} tracker status must be finite.'
				)
			return
		if not (
			self.cfg.get('flat_anchor', False)
			and self.cfg.get('flat_anchor_mode', 'residual') == 'cutie_object_only'
		):
			return
		if not isinstance(obs, TensorDict):
			raise TypeError(
				f'CutieObjectOnly {context} observation must be a TensorDict.'
			)
		keys = set(obs.keys())
		if robust_object_field.enabled(self.cfg):
			expected = robust_object_field.observation_shapes(self.cfg)
			if keys != set(expected):
				raise ValueError(
					f'ROF-WM {context} observation keys {sorted(keys)} do not '
					f'match {sorted(expected)}.'
				)
			for key, shape in expected.items():
				if tuple(obs[key].shape[-len(shape):]) != shape:
					raise ValueError(
						f'ROF-WM {context} {key} shape must end in {shape}, '
						f'got {tuple(obs[key].shape)}.'
					)
			if obs['rgb'].dtype != torch.uint8:
				raise ValueError(f'ROF-WM {context} RGB must be uint8.')
			if obs['object_mask'].dtype != torch.bool:
				raise ValueError(f'ROF-WM {context} masks must be boolean.')
			if not torch.is_floating_point(obs['object']):
				raise ValueError(f'ROF-WM {context} descriptors must be floating point.')
			if context == 'replay insertion' and not bool(
				torch.isfinite(obs['object']).all().item()
			):
				raise ValueError(f'ROF-WM {context} descriptors must be finite.')
			if obs['role_exists'].dtype != torch.float32:
				raise ValueError(f'ROF-WM {context} role_exists must be float32.')
			if not bool((obs['role_exists'] == 1).all().item()):
				raise ValueError(
					f'ROF-WM {context} forbids padding or absent role slots.'
				)
			return
		if keys != {'object'}:
			raise ValueError(
				f'CutieObjectOnly {context} observation leaked keys {sorted(keys)}.'
			)
		expected_shape = (
			int(self.cfg.get('cutie_object_num_roles', 2)),
			int(self.cfg.get('cutie_object_input_dim', 1770)),
		)
		if tuple(obs['object'].shape[-2:]) != expected_shape:
			raise ValueError(
				f'CutieObjectOnly {context} object shape must end in {expected_shape}, '
				f'got {tuple(obs["object"].shape)}.'
			)

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _validate_state_targets(self, td):
		key = object_state_supervision.REPLAY_KEY
		if not self._state_supervision_enabled:
			if key in td.keys():
				raise ValueError('Disabled run received privileged replay labels.')
			return
		if key not in td.keys():
			raise ValueError('State-supervised replay is missing same-state labels.')
		targets = td[key]
		expected = (*td.batch_size, object_state_supervision.target_dim(self.cfg))
		if tuple(targets.shape) != expected or not torch.isfinite(targets).all():
			raise ValueError(f'Invalid state targets: expected finite {expected}, got {tuple(targets.shape)}.')
		if targets.requires_grad:
			raise ValueError('Replay supervision targets must be detached labels.')

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		observations = tds.get('obs', None)
		self._validate_observation_schema(observations, 'replay insertion')
		self._observation_keys = (
			sorted(observations.keys())
			if isinstance(observations, TensorDict) else None
		)
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		self._bytes_per_step = float(bytes_per_step)
		self._storage_required_bytes = int(total_bytes)
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		if self._belief_enabled:
			# SliceSampler draws indices on the trajectory tensor's device. Bind its
			# private generator only after replay placement (CPU/CUDA) is known.
			self._belief_replay_generator = torch.Generator(
				device=self._storage_device
			)
			self._belief_replay_generator.manual_seed(
				int(self.cfg.seed)
				+ int(self.cfg.get(
					'cutie_object_belief_replay_seed_offset', 130363
				))
			)
			self._belief_sampler._rng = self._belief_replay_generator
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		self._validate_state_targets(td)
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		self._validate_state_targets(td)
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		self._validate_state_targets(td)
		td = td.select("obs", "action", "reward", "terminated", "task", object_state_supervision.REPLAY_KEY, strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		self._validate_observation_schema(obs, 'sampled batch')
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		if self._state_supervision_enabled:
			# Unlike transitions, labels retain frame zero: [H+1,B,D] aligns
			# exactly with the complete observation sequence, not action[1:].
			targets = td[object_state_supervision.REPLAY_KEY].contiguous()
			return obs, action, reward, terminated, task, targets
		return obs, action, reward, terminated, task

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)

	def sample_belief(self):
		"""Sample one long, episode-contained clean-teacher sequence batch."""
		if not self._belief_enabled:
			raise RuntimeError('Belief replay sampling is disabled for this run.')
		if self._belief_sampler is None or self._belief_replay_generator is None:
			raise RuntimeError('Belief replay sampler was not initialized.')
		# Sample indices with the isolated sampler, then read the existing lazy
		# TensorDict storage directly. This avoids both mutating the ordinary replay
		# sampler and ReplayBuffer's conflicting-batch-size warning. The buffer is
		# intentionally transform-free with LazyTensorStorage and prefetch=0.
		ordinary_sampler = self._buffer.sampler
		with self._buffer._replay_lock:
			index, _ = self._belief_sampler.sample(
				self._buffer.storage, self._belief_batch_size
			)
			td = self._buffer.storage.get(index)
		# Mirror ReplayBuffer._sample's collation step. For the configured
		# LazyTensorStorage this is identity today, while keeping this private,
		# version-pinned path faithful if TorchRL changes its default collator.
		td = self._buffer._collate_fn(td)
		if self._buffer.sampler is not ordinary_sampler:
			raise RuntimeError('Belief replay mutated the base sampler.')
		td = td.reshape(
			int(self.cfg.get(
				'cutie_object_belief_batch_size', self.cfg.batch_size
			)),
			self._belief_sequence_length,
		).permute(1, 0)
		td = td.select('obs', 'action', 'episode', strict=True).to(
			self._device, non_blocking=True
		)
		obs = td.get('obs').contiguous()
		self._validate_observation_schema(obs, 'belief sampled batch')
		episode = td.get('episode')
		if episode.shape[:2] != (
			self._belief_sequence_length,
			int(self.cfg.get('cutie_object_belief_batch_size', self.cfg.batch_size)),
		):
			raise RuntimeError(
				'Belief replay returned an unexpected sequence/batch shape: '
				f'{tuple(episode.shape)}.'
			)
		if not torch.equal(episode, episode[0:1].expand_as(episode)):
			raise RuntimeError('Belief replay slice crossed an episode boundary.')
		action = td.get('action')[1:].contiguous()
		if not torch.isfinite(action).all():
			raise RuntimeError('Belief replay contains a non-finite executed action.')
		return obs, action

import hashlib
import math as python_math

import torch
import torch.nn.functional as F

from common import math
from common import cutie_object_belief
from common import cutie_object_auxiliary
from common import cutie_proprio
from common import gt_articulated_pose
from common import multimodal_articulated_pose
from common import mask_guided_rgb
from common import visual_articulated_pose
from common import object_state_supervision
from common import robust_object_field
from common import rof_real_sibling_auxiliary
from common import rgb_interventional_auxiliary
from common import rgb_interventional_pairing
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion, LEGACY_MLP_K3_V1


ROBUST_OBJECT_FIELD_AUXILIARY_FORMAT = (
	'robust_object_field_auxiliary_ablation_v1'
)


def _finite_nonnegative_config_value(cfg, name, default):
	"""Read a finite, non-negative loss coefficient without accepting booleans."""
	value = cfg.get(name, default)
	if isinstance(value, bool):
		raise ValueError(f'{name} must be a finite non-negative number, not bool.')
	try:
		value = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(
			f'{name} must be a finite non-negative number.'
		) from exc
	if not python_math.isfinite(value) or value < 0.0:
		raise ValueError(f'{name} must be finite and non-negative.')
	return value


def robust_object_field_auxiliary_contract(cfg):
	"""Resolve the ROF-only auxiliary ablation without changing legacy defaults.

	The two ROF coefficients are multipliers on the established structured-anchor
	coefficients. Keeping both at one therefore preserves every existing run. A
	zero multiplier removes that auxiliary path while leaving the decoder, replay
	schema, controller state, and all non-ROF modes unchanged.
	"""
	reconstruction_multiplier = _finite_nonnegative_config_value(
		cfg, 'robust_object_field_auxiliary_reconstruction_coef', 1.0,
	)
	prediction_multiplier = _finite_nonnegative_config_value(
		cfg, 'robust_object_field_auxiliary_prediction_coef', 1.0,
	)
	field_enabled = robust_object_field.enabled(cfg)
	if not field_enabled:
		if reconstruction_multiplier != 1.0 or prediction_multiplier != 1.0:
			raise ValueError(
				'ROF auxiliary coefficients require '
				'robust_object_field_enabled=true.'
			)
		return None
	if (
		not cfg.get('flat_anchor', False)
		or cfg.get('flat_anchor_mode', 'residual') != 'cutie_object_only'
	):
		raise ValueError(
			'ROF auxiliary ablation requires flat_anchor_mode=cutie_object_only.'
		)
	target = str(cfg.get(
		'cutie_object_auxiliary_target',
		cutie_object_auxiliary.FULL_DESCRIPTOR,
	))
	if target not in cutie_object_auxiliary.TARGETS:
		raise ValueError(
			f'Unknown cutie_object_auxiliary_target {target!r}; expected one of '
			f'{sorted(cutie_object_auxiliary.TARGETS)}.'
		)
	base_reconstruction = _finite_nonnegative_config_value(
		cfg, 'flat_anchor_reconstruction_coef', 1.0,
	)
	base_prediction = _finite_nonnegative_config_value(
		cfg, 'flat_anchor_prediction_coef', 1.0,
	)
	return {
		'format': ROBUST_OBJECT_FIELD_AUXILIARY_FORMAT,
		'target': target,
		'query_supervision': (
			'enabled'
			if target == cutie_object_auxiliary.FULL_DESCRIPTOR
			else 'disabled'
		),
		'reconstruction_multiplier': reconstruction_multiplier,
		'prediction_multiplier': prediction_multiplier,
		'effective_reconstruction_coef': (
			base_reconstruction * reconstruction_multiplier
		),
		'effective_prediction_coef': base_prediction * prediction_multiplier,
	}


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._robust_object_field_auxiliary_contract = (
			robust_object_field_auxiliary_contract(cfg)
		)
		self._configure_real_sibling_auxiliary()
		self._configure_rgb_interventional_auxiliary()
		anchor_mode = (
			self.cfg.get('flat_anchor_mode', 'residual')
			if self.cfg.get('flat_anchor', False)
			else None
		)
		if anchor_mode == 'oc':
			# OC mode preserves the complete official RGB latent and appends one
			# sub-state per role. Parser/model-size logic still provides the RGB
			# width, so expand the joint state exactly once before model creation.
			if self.cfg.latent_dim == self.cfg.flat_anchor_scene_dim:
				self.cfg.latent_dim = self.cfg.flat_anchor_joint_dim
			elif self.cfg.latent_dim != self.cfg.flat_anchor_joint_dim:
				raise ValueError(
					'OC anchor mode expects latent_dim to equal either scene_dim '
					'or the already-expanded joint_dim.'
				)
		elif anchor_mode == 'object_graph':
			# ObjectGraph intentionally discards the RGB scene state. Its complete
			# controller state is four compact, ordered role embeddings.
			if self.cfg.flat_anchor_num_roles != 4:
				raise ValueError('ObjectGraph requires base/elbow/tip/goal roles.')
			role_dim = self.cfg.get('flat_anchor_object_role_dim', 32)
			self.cfg.latent_dim = self.cfg.flat_anchor_num_roles * role_dim
		elif anchor_mode == 'hybrid_graph':
			# WorldModel must first construct the official RGB modules at 512-D.
			# It expands cfg.latent_dim only after those exact modules are initialized.
			if self.cfg.latent_dim != self.cfg.flat_anchor_scene_dim:
				raise ValueError(
					'HybridGraph must be constructed from the official RGB latent width.'
				)
		elif anchor_mode == 'cutie_hybrid':
			# Construct every official RGB module at 512-D. WorldModel appends the
			# two 64-D Cutie object roles only after official initialization.
			if self.cfg.latent_dim != self.cfg.flat_anchor_scene_dim:
				raise ValueError(
					'CutieHybrid must be constructed from the official RGB latent width.'
				)
		elif anchor_mode == 'cutie_object_only':
			if robust_object_field.enabled(self.cfg):
				# ROF-WM packs four spatial tokens plus one ordered-query token for
				# every real role. Set the controller width before WorldModel builds
				# reward/Q/policy heads; validation below remains fail closed.
				object_dim = robust_object_field.latent_dim(self.cfg)
			else:
				# Direct variable graphs keep every configured role token; pooled
				# graphs retain the fixed-width ablation for comparison.
				variable_graph = bool(self.cfg.get(
					'cutie_object_variable_graph_enabled', False
				))
				variable_readout = str(self.cfg.get(
					'cutie_object_variable_graph_readout', 'pool'
				))
				object_dim = (
					(
						self.cfg.get('cutie_object_num_roles', 2)
						* self.cfg.get('cutie_object_role_dim', 64)
						if variable_readout == 'direct'
						else int(self.cfg.get('cutie_object_only_latent_dim', 128))
					)
					if variable_graph else
					self.cfg.get('cutie_object_num_roles', 2)
					* self.cfg.get('cutie_object_role_dim', 64)
				)
			if self.cfg.get(
				'cutie_object_only_latent_dim', 128
			) != object_dim:
				raise ValueError(
					'CutieObjectOnly configured latent width is inconsistent.'
				)
			self.cfg.latent_dim = object_dim
		elif anchor_mode == 'reward_graph':
			# RewardGraph likewise appends its graph only after every official module
			# has been constructed with the original scene width.
			if self.cfg.latent_dim != self.cfg.flat_anchor_scene_dim:
				raise ValueError(
					'RewardGraph must be constructed from the official RGB latent width.'
				)
		self.device = torch.device('cuda:0')
		self._rof_real_sibling_shift_generator = None
		if self._rof_real_sibling_aux_training:
			self._rof_real_sibling_shift_generator = torch.Generator(
				device=self.device
			)
			self._rof_real_sibling_shift_generator.manual_seed(
				int(self.cfg.seed)
				+ int(self.cfg.rof_real_sibling_aux_seed_offset) + 1
			)
		self._rgb_interventional_shift_generator = None
		self._rgb_interventional_shift_calls = 0
		self._rgb_interventional_shift_rows = 0
		self._rgb_interventional_shift_shape_trace = hashlib.sha256(
			b'rgb_interventional_augmentation_shapes_v1'
		)
		if self._rgb_interventional_aux_training:
			self._rgb_interventional_shift_generator = torch.Generator(
				device=self.device
			)
			self._rgb_interventional_shift_generator.manual_seed(
				int(self.cfg.seed)
				+ int(self.cfg.rgb_interventional_aux_seed_offset) + 1
			)
		self.model = WorldModel(cfg).to(self.device)
		self._learned_object_belief = bool(
			anchor_mode == 'cutie_object_only'
			and self.cfg.get('cutie_object_belief_enabled', False)
		)
		self._use_object_belief_for_control = bool(
			self._learned_object_belief
			and self.cfg.get('cutie_object_belief_use_for_control', False)
		)
		if self.cfg.get('cutie_object_belief_enabled', False) and not self._learned_object_belief:
			raise ValueError(
				'cutie_object_belief_enabled requires structural cutie_object_only mode.'
			)
		if self.cfg.get('cutie_object_belief_use_for_control', False) and not self._learned_object_belief:
			raise ValueError(
				'cutie_object_belief_use_for_control requires learned CutieObjectOnly belief.'
			)
		if self._learned_object_belief:
			cutie_object_belief.validate_schema(self.cfg)
			if self.cfg.get('cutie_object_last_valid_memory', False):
				raise ValueError(
					'Learned belief cannot be combined with wrapper last-valid memory.'
				)
			if self.cfg.get('cutie_object_policy_burst_plan') is not None:
				# Synthetic policy bursts are valid only for evaluation reconstructed
				# from a frozen runtime. Training runners keep this field null.
				if self.cfg.get('checkpoint') in (None, '???'):
					raise ValueError(
						'Learned-belief training requires cutie_object_policy_burst_plan=null.'
					)
		model_param_groups = [
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		]
		if anchor_mode in {'oc', 'object_graph', 'hybrid_graph', 'reward_graph'}:
			model_param_groups.append({'params': self.model._anchor_decoder.parameters()})
		elif anchor_mode in {'cutie_hybrid', 'cutie_object_only'}:
			model_param_groups.append({'params': self.model._object_decoder.parameters()})
		if anchor_mode in {'hybrid_graph', 'cutie_hybrid'}:
			model_param_groups.extend([
				{'params': self.model._graph_dynamics.parameters()},
				{'params': self.model._hybrid_reward.parameters()},
				{'params': self.model._hybrid_termination.parameters()
					if self.cfg.episodic else []},
				{'params': self.model._hybrid_q.parameters()},
			])
		elif anchor_mode == 'reward_graph':
			model_param_groups.extend([
				{'params': self.model._graph_dynamics.parameters()},
				{'params': self.model._graph_reward.parameters()},
			])
		if object_state_supervision.enabled(self.cfg):
			model_param_groups.append({
				'params': self.model._state_supervision_head.parameters(),
			})
		self.optim = torch.optim.Adam(model_param_groups, lr=self.cfg.lr, capturable=True)
		pi_params = list(self.model._pi.parameters())
		if anchor_mode in {'hybrid_graph', 'cutie_hybrid'}:
			pi_params += list(self.model._hybrid_pi.parameters())
		self.pi_optim = torch.optim.Adam(pi_params, lr=self.cfg.lr, eps=1e-5, capturable=True)
		if self._learned_object_belief:
			belief_params = tuple(self.model._belief_dynamics.parameters())
			if not belief_params:
				raise RuntimeError('Learned belief transition has no trainable parameters.')
			model_optim_ids = {
				id(parameter)
				for group in self.optim.param_groups
				for parameter in group['params']
			}
			if model_optim_ids & {id(parameter) for parameter in belief_params}:
				raise RuntimeError(
					'Belief transition parameters leaked into the base world optimizer.'
				)
			self.belief_optim = torch.optim.Adam(
				belief_params,
				lr=float(self.cfg.get('cutie_object_belief_lr', self.cfg.lr)),
				capturable=True,
			)
			self._belief_mask_generator = torch.Generator(device='cpu')
			self._belief_mask_generator.manual_seed(
				int(self.cfg.seed)
				+ int(self.cfg.get('cutie_object_belief_mask_seed_offset', 104729))
			)
			self._belief_update_calls = 0
			self._belief_aux_updates = 0
		else:
			self.belief_optim = None
			self._belief_mask_generator = None
			self._belief_update_calls = 0
			self._belief_aux_updates = 0
		self._belief_aux_attempts = 0
		self._belief_aux_no_teacher_skips = 0
		self._belief_age20_available_updates = 0
		self._belief_age20_teacher_roles = [0, 0]
		self._belief_reacquisition_available_updates = 0
		self._belief_reacquisition_teacher_roles = [0, 0]
		self._configure_gradient_partitions(anchor_mode)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		self._online_object_belief = None
		self._online_object_belief_action = None
		self._online_object_belief_prior_uses = 0
		self._online_object_belief_resets = 0
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			if self.cfg.get('compile_fallback_random', False):
				# Torch 2.7 forbids passing both mode and options. Preserve the
				# reduce-overhead CUDA-graph option explicitly while routing random
				# ops through eager CUDA RNG for cross-graph reproducibility.
				self._update = torch.compile(self._update, options={
					'fallback_random': True,
					'triton.cudagraphs': True,
				})
			else:
				self._update = torch.compile(self._update, mode="reduce-overhead")

	@staticmethod
	def _module_parameters(*modules):
		return tuple(
			parameter
			for module in modules
			if module is not None
			for parameter in module.parameters()
		)

	def _configure_real_sibling_auxiliary(self):
		"""Bind the opt-in training capsule without making evaluation read it.

		Training validates every immutable shard before a GPU model is created and
		writes the resulting manifest identity back into the runtime config.  A
		pure-policy evaluator reconstructs the same checkpoint contract from that
		identity alone; it neither opens nor retains the auxiliary training data.
		"""
		self._rof_real_sibling_aux_enabled = rof_real_sibling_auxiliary.enabled(
			self.cfg
		)
		self._rof_real_sibling_aux_training = bool(
			self._rof_real_sibling_aux_enabled
			and self.cfg.get('checkpoint', '???') in (None, '???')
		)
		self._rof_real_sibling_aux_replay = None
		self._rof_real_sibling_aux_attempts = 0
		self._rof_real_sibling_aux_updates = 0
		self._rof_real_sibling_aux_samples = 0
		self._rof_real_sibling_aux_contract = None
		if not self._rof_real_sibling_aux_enabled:
			return
		if self._rof_real_sibling_aux_training:
			manifest_path = self.cfg.get('rof_real_sibling_aux_manifest', None)
			if manifest_path is None:
				raise ValueError(
					'Enabled real-sibling training requires its materialized manifest.'
				)
			payload = rof_real_sibling_auxiliary.validate_manifest(
				manifest_path, self.cfg
			)
			self.cfg.rof_real_sibling_aux_manifest_sha256 = payload[
				'_manifest_sha256'
			]
			self.cfg.rof_real_sibling_aux_capsule_format = payload['format']
			self.cfg.rof_real_sibling_aux_source_dataset_sha256 = payload[
				'source_dataset'
			]['manifest_sha256']
		# Evaluation is deliberately independent of the training capsule path.  It
		# must still carry the immutable SHA/format copied from runtime_config so
		# checkpoint provenance cannot silently change.
		self._rof_real_sibling_aux_contract = (
			rof_real_sibling_auxiliary.validate_config(
				self.cfg, require_bound_manifest=True
			)
		)

	def _configure_rgb_interventional_auxiliary(self):
		"""Bind a raw-RGB intervention capsule only for model training.

		Evaluation reconstructs the immutable contract from scalar identities in
		the runtime config and checkpoint.  It never opens the training capsule.
		"""
		self._rgb_interventional_aux_enabled = (
			rgb_interventional_auxiliary.enabled(self.cfg)
		)
		if self._rgb_interventional_aux_enabled and self._rof_real_sibling_aux_enabled:
			raise ValueError(
				'Raw-RGB and ROF real-sibling auxiliaries are mutually exclusive.'
			)
		self._rgb_interventional_aux_training = bool(
			self._rgb_interventional_aux_enabled
			and self.cfg.get('checkpoint', '???') in (None, '???')
		)
		self._rgb_interventional_aux_replay = None
		self._rgb_interventional_aux_attempts = 0
		self._rgb_interventional_aux_updates = 0
		self._rgb_interventional_aux_samples = 0
		self._rgb_interventional_aux_contract = None
		self._rgb_interventional_pairing_schedule = None
		if not self._rgb_interventional_aux_enabled:
			# Even a disabled run must not silently carry stale capsule identity.
			# This is validation-only and does not allocate replay state or draw RNG.
			rgb_interventional_auxiliary.validate_config(self.cfg)
			return
		if self._rgb_interventional_aux_training:
			manifest_path = self.cfg.get('rgb_interventional_aux_manifest', None)
			if manifest_path is None:
				raise ValueError(
					'Enabled RGB interventional training requires its manifest.'
				)
			payload = rgb_interventional_auxiliary.validate_manifest(
				manifest_path, self.cfg
			)
			self.cfg.rgb_interventional_aux_manifest_sha256 = payload[
				'_manifest_sha256'
			]
			self.cfg.rgb_interventional_aux_capsule_format = payload['format']
			self.cfg.rgb_interventional_aux_source_dataset_sha256 = payload[
				'source_dataset'
			]['manifest_sha256']
		self._rgb_interventional_aux_contract = (
			rgb_interventional_auxiliary.validate_config(
				self.cfg, require_bound_manifest=True
			)
		)
		if self._rgb_interventional_aux_training:
			pairing = self._rgb_interventional_aux_contract['pairing']
			self._rgb_interventional_pairing_schedule = (
				rgb_interventional_pairing.Schedule(
					seed=int(pairing['effective_seed']),
					batch_size=int(pairing['batch_size']),
					mode=str(pairing['mode']),
				)
			)

	def _configure_gradient_partitions(self, anchor_mode):
		"""Prevent graph auxiliary gradients from clipping official RGB updates."""
		self._hybrid_graph = anchor_mode in {'hybrid_graph', 'cutie_hybrid'}
		self._cutie_hybrid = anchor_mode == 'cutie_hybrid'
		self._cutie_object_only = anchor_mode == 'cutie_object_only'
		self._cutie_object_mode = self._cutie_hybrid or self._cutie_object_only
		self._reward_graph = anchor_mode == 'reward_graph'
		if self._hybrid_graph:
			self._official_world_params = self._module_parameters(
				self.model._encoder['rgb'],
				self.model._dynamics,
				self.model._reward,
				self.model._termination,
				self.model._Qs,
			)
			object_encoder = self.model._encoder[
				'object' if self._cutie_hybrid else 'anchor'
			]
			object_decoder = (
				self.model._object_decoder
				if self._cutie_hybrid else self.model._anchor_decoder
			)
			self._graph_world_params = self._module_parameters(
				object_encoder,
				self.model._graph_dynamics,
				object_decoder,
				self.model._hybrid_reward,
				self.model._hybrid_termination,
				self.model._hybrid_q,
			)
			self._official_pi_params = self._module_parameters(self.model._pi)
			self._graph_pi_params = self._module_parameters(self.model._hybrid_pi)
		elif self._reward_graph:
			self._official_world_params = self._module_parameters(
				self.model._encoder['rgb'],
				self.model._dynamics,
				self.model._reward,
				self.model._termination,
				self.model._Qs,
			)
			# Keep high-weight graph consistency/reconstruction gradients in their
			# own clip group. The graph reward head has an independent norm budget.
			self._graph_aux_params = self._module_parameters(
				self.model._encoder['anchor'],
				self.model._graph_dynamics,
				self.model._anchor_decoder,
			)
			self._graph_reward_params = self._module_parameters(
				self.model._graph_reward
			)

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			if self.cfg.get('compile_fallback_random', False):
				plan = torch.compile(self._plan, options={
					'fallback_random': True,
					'triton.cudagraphs': True,
				})
			else:
				plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	@property
	def belief_plan(self):
		"""Return the latent-input planner used by the online learned belief."""
		if not self._learned_object_belief:
			raise RuntimeError('Latent belief planning is disabled for this agent.')
		plan = getattr(self, '_belief_plan_val', None)
		if plan is not None:
			return plan
		if self.cfg.compile:
			if self.cfg.get('compile_fallback_random', False):
				plan = torch.compile(self._plan_from_latent, options={
					'fallback_random': True,
					'triton.cudagraphs': True,
				})
			else:
				plan = torch.compile(
					self._plan_from_latent, mode='reduce-overhead'
				)
		else:
			plan = self._plan_from_latent
		self._belief_plan_val = plan
		return plan

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		if self._learned_object_belief and self._belief_aux_updates < 1:
			raise RuntimeError(
				'Learned-belief checkpoint cannot be saved before a successful '
				'belief auxiliary update.'
			)
		if self._learned_object_belief and (
			min(self._belief_age20_teacher_roles) < 1
			or min(self._belief_reacquisition_teacher_roles) < 1
		):
				raise RuntimeError(
					'Learned-belief checkpoint requires cumulative clean supervision '
						'for both roles at age 20 and reacquisition.'
				)
		if (
			self._rof_real_sibling_aux_enabled
			and self._rof_real_sibling_aux_updates < 1
		):
			raise RuntimeError(
				'Real-sibling checkpoint cannot be saved before a successful '
				'auxiliary update.'
			)
		if (
			self._rgb_interventional_aux_enabled
			and self._rgb_interventional_aux_updates < 1
		):
			raise RuntimeError(
				'RGB interventional checkpoint cannot be saved before a successful '
				'auxiliary update.'
			)
		contract = {
			'format': 'tdmpc2_checkpoint_contract_v1',
			'flat_anchor_mode': str(self.cfg.get('flat_anchor_mode', 'residual')),
			'latent_dim': int(self.cfg.latent_dim),
			'cutie_object_belief_enabled': bool(self._learned_object_belief),
			'cutie_object_belief_aux_updates': int(self._belief_aux_updates),
		}
		if robust_object_field.enabled(self.cfg):
			contract['robust_object_field'] = robust_object_field.contract(self.cfg)
			contract['robust_object_field_auxiliary'] = dict(
				self._robust_object_field_auxiliary_contract
			)
		if self._rof_real_sibling_aux_enabled:
			replay_metrics = (
				self._rof_real_sibling_aux_replay.metrics
				if self._rof_real_sibling_aux_replay is not None else {}
			)
			contract['rof_real_sibling_auxiliary'] = {
				'contract': dict(self._rof_real_sibling_aux_contract),
				'supervision': {
					'attempts': int(self._rof_real_sibling_aux_attempts),
					'successful_updates': int(self._rof_real_sibling_aux_updates),
					'sampled_root_pairs': int(
						self._rof_real_sibling_aux_samples
					),
					'sampler_metrics': replay_metrics,
				},
			}
		if self._rgb_interventional_aux_enabled:
			replay_metrics = (
				self._rgb_interventional_aux_replay.metrics
				if self._rgb_interventional_aux_replay is not None else {}
			)
			augmentation_metrics = {}
			pairing_metrics = (
				self._rgb_interventional_pairing_schedule.metrics
				if self._rgb_interventional_pairing_schedule is not None else {}
			)
			if self._rgb_interventional_shift_generator is not None:
				generator_state = (
					self._rgb_interventional_shift_generator.get_state()
					.detach().cpu().numpy().tobytes()
				)
				augmentation_metrics = {
					'crop_calls': int(self._rgb_interventional_shift_calls),
					'crop_rows': int(self._rgb_interventional_shift_rows),
					'draw_shape_sequence_sha256': (
						self._rgb_interventional_shift_shape_trace.hexdigest()
					),
					'generator_state_sha256': hashlib.sha256(
						generator_state
					).hexdigest(),
				}
			contract['rgb_interventional_auxiliary'] = {
				'contract': dict(self._rgb_interventional_aux_contract),
				'supervision': {
					'attempts': int(self._rgb_interventional_aux_attempts),
					'successful_updates': int(
						self._rgb_interventional_aux_updates
					),
					'sampled_root_pairs': int(
						self._rgb_interventional_aux_samples
					),
					'sampler_metrics': replay_metrics,
					'augmentation_metrics': augmentation_metrics,
					'pairing_metrics': pairing_metrics,
				},
			}
		if mask_guided_rgb.enabled(self.cfg):
			contract['mask_guided_rgb'] = dict(
				mask_guided_rgb.contract(self.cfg)
			)
		regression_encoder = self.cfg.get('cutie_object_regression_encoder', None)
		if regression_encoder is not None:
			contract['cutie_object_regression_encoder'] = regression_encoder
		if self._cutie_object_mode:
			contract['cutie_object_auxiliary'] = dict(
				self.model._cutie_object_auxiliary_contract
			)
		if self._cutie_object_only:
			variant = str(self.cfg.get('cutie_object_observation_variant', 'full'))
			frame_schema = str(self.cfg.get(
				'cutie_object_frame_schema', 'cutie_query_mask_status_v1'
			))
			contract['cutie_object_observation'] = {
				'format': 'cutie_object_observation_contract_v1',
				'variant': variant,
				'frame_schema': frame_schema,
				'privileged_runtime_segmentation': bool(self.cfg.get(
					'cutie_object_allow_simulator_runtime', False
				)),
				'num_roles': int(self.cfg.get('cutie_object_num_roles', 2)),
				'frame_dim': int(self.cfg.get('cutie_object_frame_dim', 590)),
				'stack_frames': int(self.cfg.get('cutie_object_stack_frames', 3)),
				'input_dim': int(self.cfg.get('cutie_object_input_dim', 1770)),
				'spatial_token_enabled': bool(self.cfg.get(
					'cutie_object_spatial_token_enabled', False
				)),
				'spatial_graph_path': self.cfg.get(
					'cutie_object_spatial_graph_path', None
				),
				'spatial_token_dim': int(self.cfg.get(
					'cutie_object_spatial_token_dim', 64
				)),
				'spatial_num_heads': int(self.cfg.get(
					'cutie_object_spatial_num_heads', 4
				)),
				'spatial_num_layers': int(self.cfg.get(
					'cutie_object_spatial_num_layers', 2
				)),
			}
			if self.cfg.get('cutie_object_variable_graph_enabled', False):
				contract['cutie_object_observation'].update({
					'variable_graph_enabled': True,
					'variable_graph_max_roles': int(self.cfg.get(
						'cutie_object_variable_graph_max_roles', 8
					)),
					'variable_graph_pool_tokens': int(self.cfg.get(
						'cutie_object_variable_graph_pool_tokens', 2
					)),
					'variable_graph_primary_skip_enabled': bool(self.cfg.get(
						'cutie_object_variable_graph_primary_skip_enabled', False
					)),
					'variable_graph_readout': str(self.cfg.get(
						'cutie_object_variable_graph_readout', 'pool'
					)),
				})
			if gt_articulated_pose.enabled(self.cfg):
				contract['gt_articulated_pose_observation'] = (
					gt_articulated_pose.observation_contract(self.cfg)
				)
			if visual_articulated_pose.enabled(self.cfg):
				contract['visual_articulated_pose_observation'] = (
					visual_articulated_pose.observation_contract(self.cfg)
				)
			if multimodal_articulated_pose.enabled(self.cfg):
				contract['multimodal_articulated_pose_observation'] = (
					multimodal_articulated_pose.observation_contract(self.cfg)
				)
			if cutie_proprio.enabled(self.cfg):
				contract['cutie_proprio_observation'] = (
					cutie_proprio.observation_contract(self.cfg)
				)
		if self._learned_object_belief:
			contract['cutie_object_belief_use_for_control_during_training'] = bool(
				self.cfg.get('cutie_object_belief_use_for_control', False)
			)
			contract['cutie_object_belief_collection_mode'] = (
				'online_belief' if contract[
					'cutie_object_belief_use_for_control_during_training'
				] else 'measurement_only_shadow'
			)
			contract['cutie_object_belief_online_control_before_checkpoint_save'] = {
				'prior_role_uses': int(self._online_object_belief_prior_uses),
				'episode_state_resets': int(self._online_object_belief_resets),
			}
			contract['cutie_object_belief_schema'] = {
				'num_roles': int(self.cfg.get('cutie_object_num_roles', 2)),
				'frame_dim': int(self.cfg.get('cutie_object_frame_dim', 590)),
				'stack_frames': int(self.cfg.get('cutie_object_stack_frames', 3)),
				'input_dim': int(self.cfg.get('cutie_object_input_dim', 1770)),
				'role_dim': int(self.cfg.get('cutie_object_role_dim', 64)),
			}
			contract['cutie_object_belief_training'] = {
				'batch_size': int(self.cfg.get(
					'cutie_object_belief_batch_size', self.cfg.batch_size
				)),
				'burn_in': int(self.cfg.get('cutie_object_belief_burn_in', 3)),
				'min_burst': int(self.cfg.get('cutie_object_belief_min_burst', 5)),
				'max_burst': int(self.cfg.get('cutie_object_belief_max_burst', 20)),
				'recovery_frames': int(self.cfg.get(
					'cutie_object_belief_recovery_frames', 1
				)),
				'update_frequency': int(self.cfg.get(
					'cutie_object_belief_update_frequency', 4
				)),
				'loss_coef': float(self.cfg.get(
					'cutie_object_belief_loss_coef', 1.0
				)),
				'reacquisition_coef': float(self.cfg.get(
					'cutie_object_belief_reacquisition_coef', 1.0
				)),
			}
			contract['cutie_object_belief_supervision'] = {
				'format': 'cutie_object_belief_supervision_v1',
				'attempts': int(self._belief_aux_attempts),
				'successful_updates': int(self._belief_aux_updates),
				'no_teacher_skips': int(self._belief_aux_no_teacher_skips),
				'age20_available_updates': int(
					self._belief_age20_available_updates
				),
				'age20_teacher_roles': [
					int(value) for value in self._belief_age20_teacher_roles
				],
				'reacquisition_available_updates': int(
					self._belief_reacquisition_available_updates
				),
				'reacquisition_teacher_roles': [
					int(value)
					for value in self._belief_reacquisition_teacher_roles
				],
			}
		if object_state_supervision.enabled(self.cfg):
			contract['object_state_supervision'] = object_state_supervision.contract(self.cfg)
		if self.cfg.get('cutie_object_true_entity_enabled', False):
			contract['cutie_true_entity_input'] = 'single_entity_raw_mask_v1'
		torch.save({
			"model": self.model.state_dict(),
			"checkpoint_contract": contract,
		}, fp)

	def load(self, fp, allow_official_warmstart=False):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
			allow_official_warmstart (bool): Explicitly allow a pure official RGB
				checkpoint to initialize only a graph mode's shared scene path.
		"""
		if isinstance(fp, dict):
			payload = fp
		else:
			payload = torch.load(
				fp, map_location=torch.get_default_device(), weights_only=False
			)
		contract = (
			payload.get('checkpoint_contract')
			if isinstance(payload, dict) else None
		)
		source_regression = (
			contract.get('cutie_object_regression_encoder')
			if isinstance(contract, dict) else None
		)
		expected_regression = self.cfg.get('cutie_object_regression_encoder', None)
		if source_regression != expected_regression:
			raise RuntimeError(
				'Cutie regression encoder checkpoint contract mismatch: '
				f'{source_regression!r} != {expected_regression!r}.'
			)
		source_state_supervision = (
			contract.get('object_state_supervision') if isinstance(contract, dict) else None
		)
		expected_state_supervision = (
			object_state_supervision.contract(self.cfg)
			if object_state_supervision.enabled(self.cfg) else None
		)
		if source_state_supervision != expected_state_supervision:
			raise RuntimeError('Training-only state supervision checkpoint contract mismatch.')
		source_object_field = (
			contract.get('robust_object_field') if isinstance(contract, dict) else None
		)
		expected_object_field = (
			robust_object_field.contract(self.cfg)
			if robust_object_field.enabled(self.cfg) else None
		)
		if source_object_field != expected_object_field:
			raise RuntimeError(
				'ROF-WM checkpoint contract mismatch: '
				f'{source_object_field!r} != {expected_object_field!r}.'
			)
		source_mask_guided = (
			contract.get('mask_guided_rgb') if isinstance(contract, dict) else None
		)
		expected_mask_guided = (
			dict(mask_guided_rgb.contract(self.cfg))
			if mask_guided_rgb.enabled(self.cfg) else None
		)
		if source_mask_guided != expected_mask_guided:
			raise RuntimeError(
				'Mask-guided RGB checkpoint contract mismatch: '
				f'{source_mask_guided!r} != {expected_mask_guided!r}.'
			)
		source_field_auxiliary = (
			contract.get('robust_object_field_auxiliary')
			if isinstance(contract, dict) else None
		)
		expected_field_auxiliary = (
			self._robust_object_field_auxiliary_contract
		)
		if expected_field_auxiliary is None:
			if source_field_auxiliary is not None:
				raise RuntimeError(
					'ROF auxiliary checkpoint contract is present for a non-ROF model.'
				)
		elif source_field_auxiliary is None:
			# ROF-WM V0 checkpoints predate explicit ROF-only multipliers. They
			# therefore mean the established value one; the existing Cutie
			# auxiliary contract below still validates the exact target schema.
			if (
				expected_field_auxiliary['reconstruction_multiplier'] != 1.0
				or expected_field_auxiliary['prediction_multiplier'] != 1.0
			):
				raise RuntimeError(
					'Legacy ROF checkpoint cannot initialize a non-default auxiliary '
					'coefficient ablation.'
				)
		elif source_field_auxiliary != expected_field_auxiliary:
				raise RuntimeError(
					'ROF auxiliary checkpoint contract mismatch: '
						f'{source_field_auxiliary!r} != {expected_field_auxiliary!r}.'
				)
		source_sibling_auxiliary = (
			contract.get('rof_real_sibling_auxiliary')
			if isinstance(contract, dict) else None
		)
		expected_sibling_auxiliary = self._rof_real_sibling_aux_contract
		sibling_supervision = (
			rof_real_sibling_auxiliary.validate_checkpoint_record(
				source_sibling_auxiliary, expected_sibling_auxiliary
			)
		)
		self._rof_real_sibling_aux_attempts = sibling_supervision['attempts']
		self._rof_real_sibling_aux_updates = sibling_supervision[
			'successful_updates'
		]
		self._rof_real_sibling_aux_samples = sibling_supervision[
			'sampled_root_pairs'
		]
		source_rgb_interventional = (
			contract.get('rgb_interventional_auxiliary')
			if isinstance(contract, dict) else None
		)
		rgb_interventional_supervision = (
			rgb_interventional_auxiliary.validate_checkpoint_record(
				source_rgb_interventional,
				self._rgb_interventional_aux_contract,
			)
		)
		self._rgb_interventional_aux_attempts = (
			rgb_interventional_supervision['attempts']
		)
		self._rgb_interventional_aux_updates = (
			rgb_interventional_supervision['successful_updates']
		)
		self._rgb_interventional_aux_samples = (
			rgb_interventional_supervision['sampled_root_pairs']
		)
		source_entity = contract.get('cutie_true_entity_input') if isinstance(contract, dict) else None
		expected_entity = (
			'single_entity_raw_mask_v1'
			if self.cfg.get('cutie_object_true_entity_enabled', False) else None
		)
		if source_entity != expected_entity:
			raise RuntimeError('Single-entity tracking checkpoint contract mismatch.')
		if self._cutie_object_only:
			expected_variant = str(self.cfg.get(
				'cutie_object_observation_variant', 'full'
			))
			expected_observation = {
				'format': 'cutie_object_observation_contract_v1',
				'variant': expected_variant,
				'frame_schema': str(self.cfg.get(
					'cutie_object_frame_schema', 'cutie_query_mask_status_v1'
				)),
				'privileged_runtime_segmentation': bool(self.cfg.get(
					'cutie_object_allow_simulator_runtime', False
				)),
				'num_roles': int(self.cfg.get('cutie_object_num_roles', 2)),
				'frame_dim': int(self.cfg.get('cutie_object_frame_dim', 590)),
				'stack_frames': int(self.cfg.get('cutie_object_stack_frames', 3)),
				'input_dim': int(self.cfg.get('cutie_object_input_dim', 1770)),
				'spatial_token_enabled': bool(self.cfg.get(
					'cutie_object_spatial_token_enabled', False
				)),
				'spatial_graph_path': self.cfg.get(
					'cutie_object_spatial_graph_path', None
				),
				'spatial_token_dim': int(self.cfg.get(
					'cutie_object_spatial_token_dim', 64
				)),
				'spatial_num_heads': int(self.cfg.get(
					'cutie_object_spatial_num_heads', 4
				)),
				'spatial_num_layers': int(self.cfg.get(
					'cutie_object_spatial_num_layers', 2
				)),
			}
			if self.cfg.get('cutie_object_variable_graph_enabled', False):
				expected_observation.update({
					'variable_graph_enabled': True,
					'variable_graph_max_roles': int(self.cfg.get(
						'cutie_object_variable_graph_max_roles', 8
					)),
					'variable_graph_pool_tokens': int(self.cfg.get(
						'cutie_object_variable_graph_pool_tokens', 2
					)),
					'variable_graph_primary_skip_enabled': bool(self.cfg.get(
						'cutie_object_variable_graph_primary_skip_enabled', False
					)),
					'variable_graph_readout': str(self.cfg.get(
						'cutie_object_variable_graph_readout', 'pool'
					)),
				})
			source_observation = (
				contract.get('cutie_object_observation')
				if isinstance(contract, dict) else None
			)
			spatial_contract_keys = {
				'spatial_token_enabled', 'spatial_graph_path', 'spatial_token_dim',
				'spatial_num_heads', 'spatial_num_layers', 'variable_graph_enabled',
				'variable_graph_max_roles', 'variable_graph_pool_tokens',
				'variable_graph_primary_skip_enabled', 'variable_graph_readout',
			}
			if (
				source_observation is None
				and expected_variant == 'full'
				and not expected_observation['spatial_token_enabled']
				and expected_regression != LEGACY_MLP_K3_V1
			):
				# Legacy Cutie ObjectOnly checkpoints predate an explicit schema
				# contract. They can only mean the established full query+mask path.
				pass
			elif (
				isinstance(source_observation, dict)
				and not expected_observation['spatial_token_enabled']
				and spatial_contract_keys.isdisjoint(source_observation)
				and expected_regression != LEGACY_MLP_K3_V1
			):
				legacy_expected = {
					key: value for key, value in expected_observation.items()
					if key not in spatial_contract_keys
				}
				if source_observation != legacy_expected:
					raise RuntimeError(
						'Cutie ObjectOnly legacy observation contract mismatch: '
						f'{source_observation!r} != {legacy_expected!r}.'
					)
			elif source_observation != expected_observation:
				raise RuntimeError(
					'Cutie ObjectOnly observation contract mismatch: '
					f'{source_observation!r} != {expected_observation!r}.'
				)
			if gt_articulated_pose.enabled(self.cfg):
				expected_pose = gt_articulated_pose.observation_contract(self.cfg)
				source_pose = (
					contract.get('gt_articulated_pose_observation')
					if isinstance(contract, dict) else None
				)
				if source_pose != expected_pose:
					raise RuntimeError(
						'GT articulated-pose observation contract mismatch: '
						f'{source_pose!r} != {expected_pose!r}.'
					)
			if visual_articulated_pose.enabled(self.cfg):
				expected_pose = visual_articulated_pose.observation_contract(self.cfg)
				source_pose = (
					contract.get('visual_articulated_pose_observation')
					if isinstance(contract, dict) else None
				)
				if source_pose != expected_pose:
					raise RuntimeError(
						'Visual articulated-pose observation contract mismatch: '
						f'{source_pose!r} != {expected_pose!r}.'
					)
			if multimodal_articulated_pose.enabled(self.cfg):
				expected_pose = multimodal_articulated_pose.observation_contract(
					self.cfg
				)
				source_pose = (
					contract.get('multimodal_articulated_pose_observation')
					if isinstance(contract, dict) else None
				)
				if source_pose != expected_pose:
					raise RuntimeError(
						'Multimodal articulated-pose observation contract mismatch: '
						f'{source_pose!r} != {expected_pose!r}.'
					)
			if cutie_proprio.enabled(self.cfg):
				expected_pose = cutie_proprio.observation_contract(self.cfg)
				source_pose = (
					contract.get('cutie_proprio_observation')
					if isinstance(contract, dict) else None
				)
				if source_pose != expected_pose:
					raise RuntimeError(
						'Cutie-proprio observation contract mismatch: '
						f'{source_pose!r} != {expected_pose!r}.'
					)
		if self._cutie_object_mode:
			expected_auxiliary = self.model._cutie_object_auxiliary_contract
			source_auxiliary = (
				contract.get('cutie_object_auxiliary')
				if isinstance(contract, dict) else None
			)
			source_mode = (
				contract.get('flat_anchor_mode')
				if isinstance(contract, dict) else None
			)
			raw_source_state = (
				payload.get('model', payload)
				if isinstance(payload, dict) else payload
			)
			source_has_cutie_object_branch = (
				isinstance(raw_source_state, dict)
				and any(
					str(key).startswith('_encoder.object.')
					for key in raw_source_state
				)
			)
			is_official_hybrid_warmstart = (
				self._cutie_hybrid
				and allow_official_warmstart
				and source_mode not in {'cutie_hybrid', 'cutie_object_only'}
				and not source_has_cutie_object_branch
			)
			if source_auxiliary is None and not is_official_hybrid_warmstart:
				# Checkpoints predating this field used the frozen default objective.
				source_observation = (
					contract.get('cutie_object_observation')
					if isinstance(contract, dict) else None
				)
				source_auxiliary = cutie_object_auxiliary.legacy_contract(
					source_observation
				)
			if (
				not is_official_hybrid_warmstart
				and source_auxiliary != expected_auxiliary
			):
				raise RuntimeError(
					'Cutie object auxiliary contract mismatch: '
					f'{source_auxiliary!r} != {expected_auxiliary!r}.'
				)
		if self._learned_object_belief:
			expected_schema = {
				'num_roles': int(self.cfg.get('cutie_object_num_roles', 2)),
				'frame_dim': int(self.cfg.get('cutie_object_frame_dim', 590)),
				'stack_frames': int(self.cfg.get('cutie_object_stack_frames', 3)),
				'input_dim': int(self.cfg.get('cutie_object_input_dim', 1770)),
				'role_dim': int(self.cfg.get('cutie_object_role_dim', 64)),
			}
			expected_training = {
				'batch_size': int(self.cfg.get(
					'cutie_object_belief_batch_size', self.cfg.batch_size
				)),
				'burn_in': int(self.cfg.get('cutie_object_belief_burn_in', 3)),
				'min_burst': int(self.cfg.get('cutie_object_belief_min_burst', 5)),
				'max_burst': int(self.cfg.get('cutie_object_belief_max_burst', 20)),
				'recovery_frames': int(self.cfg.get(
					'cutie_object_belief_recovery_frames', 1
				)),
				'update_frequency': int(self.cfg.get(
					'cutie_object_belief_update_frequency', 4
				)),
				'loss_coef': float(self.cfg.get(
					'cutie_object_belief_loss_coef', 1.0
				)),
				'reacquisition_coef': float(self.cfg.get(
					'cutie_object_belief_reacquisition_coef', 1.0
				)),
			}
			expected = {
				'format': 'tdmpc2_checkpoint_contract_v1',
				'flat_anchor_mode': 'cutie_object_only',
				'latent_dim': int(self.cfg.latent_dim),
				'cutie_object_belief_enabled': True,
				'cutie_object_belief_schema': expected_schema,
				'cutie_object_belief_training': expected_training,
			}
			if not isinstance(contract, dict):
				raise RuntimeError(
					'Learned-belief checkpoint is missing its training contract.'
				)
			bad = {
				key: (contract.get(key), value)
				for key, value in expected.items()
				if contract.get(key) != value
			}
			aux_updates = contract.get('cutie_object_belief_aux_updates')
			if isinstance(aux_updates, bool) or not isinstance(aux_updates, int):
				bad['cutie_object_belief_aux_updates'] = (aux_updates, 'positive int')
			elif aux_updates < 1:
				bad['cutie_object_belief_aux_updates'] = (aux_updates, '>=1')
			if bad:
				raise RuntimeError(
					'Learned-belief checkpoint contract mismatch: '
					f'{bad}.'
				)
			training_control = contract.get(
				'cutie_object_belief_use_for_control_during_training'
			)
			collection_mode = contract.get('cutie_object_belief_collection_mode')
			if training_control is None and collection_mode is None:
				# v4 checkpoints predate the explicit control split and always used
				# the learned posterior for collection and training-time evaluation.
				self._use_object_belief_for_control = True
			elif (
				not isinstance(training_control, bool)
				or collection_mode not in {
					'online_belief', 'measurement_only_shadow'
				}
				or (collection_mode == 'online_belief') != training_control
			):
				raise RuntimeError(
					'Learned-belief checkpoint collection-mode metadata is invalid.'
				)
		state_dict = payload["model"] if "model" in payload else payload
		target_state = self.model.state_dict()
		state_dict = api_model_conversion(target_state, state_dict)
		if self._hybrid_graph:
			if self._cutie_hybrid:
				hybrid_prefixes = (
					'_encoder.object.',
					'_graph_dynamics.',
					'_object_decoder.',
					'_hybrid_reward.',
					'_hybrid_pi.',
					'_hybrid_termination.',
					'_hybrid_q.',
					'_detach_hybrid_q_params.',
					'_target_hybrid_q_params.',
				)
			else:
				hybrid_prefixes = (
					'_encoder.anchor.',
					'_graph_dynamics.',
					'_anchor_decoder.',
					'_hybrid_reward.',
					'_hybrid_pi.',
					'_hybrid_termination.',
					'_hybrid_q.',
					'_detach_hybrid_q_params.',
					'_target_hybrid_q_params.',
				)
			source_has_hybrid = any(
				key.startswith(hybrid_prefixes) for key in state_dict
			)
			if not source_has_hybrid and allow_official_warmstart:
				missing = set(target_state) - set(state_dict)
				unexpected = set(state_dict) - set(target_state)
				expected_missing = {
					key for key in target_state if key.startswith(hybrid_prefixes)
				}
				if unexpected or missing != expected_missing:
					raise RuntimeError(
						'Official-to-HybridGraph checkpoint conversion found '
						f'unexpected={sorted(unexpected)[:10]} and '
						f'non-hybrid missing={sorted(missing - expected_missing)[:10]}.'
					)
				# Warm-start every official tensor and retain the freshly initialized,
				# identity-preserving graph branch. Hybrid checkpoints remain strict.
				state_dict = dict(state_dict)
				state_dict.update({key: target_state[key] for key in expected_missing})
		elif self._reward_graph:
			reward_graph_prefixes = (
				'_encoder.anchor.',
				'_graph_dynamics.',
				'_anchor_decoder.',
				'_graph_reward.',
			)
			source_has_reward_graph = any(
				key.startswith(reward_graph_prefixes) for key in state_dict
			)
			if not source_has_reward_graph and allow_official_warmstart:
				missing = set(target_state) - set(state_dict)
				unexpected = set(state_dict) - set(target_state)
				expected_missing = {
					key for key in target_state if key.startswith(reward_graph_prefixes)
				}
				if unexpected or missing != expected_missing:
					raise RuntimeError(
						'Official-to-RewardGraph checkpoint conversion found '
						f'unexpected={sorted(unexpected)[:10]} and '
						f'non-graph missing={sorted(missing - expected_missing)[:10]}.'
					)
				state_dict = dict(state_dict)
				state_dict.update({key: target_state[key] for key in expected_missing})
		self.model.load_state_dict(state_dict)
		if self._learned_object_belief:
			self._belief_aux_updates = int(
				contract['cutie_object_belief_aux_updates']
			)
			supervision = contract.get('cutie_object_belief_supervision')
			if supervision is not None:
				if not isinstance(supervision, dict) or supervision.get(
					'format'
				) != 'cutie_object_belief_supervision_v1':
					raise RuntimeError(
						'Learned-belief checkpoint supervision metadata is invalid.'
					)
				age20_roles = supervision.get('age20_teacher_roles')
				reacquisition_roles = supervision.get(
					'reacquisition_teacher_roles'
				)
				integer_fields = (
					'attempts', 'successful_updates', 'no_teacher_skips',
					'age20_available_updates',
					'reacquisition_available_updates',
				)
				if (
					any(
						isinstance(supervision.get(key), bool)
						or not isinstance(supervision.get(key), int)
						or supervision.get(key) < 0
						for key in integer_fields
					)
					or not isinstance(age20_roles, list)
					or not isinstance(reacquisition_roles, list)
					or len(age20_roles) != 2
					or len(reacquisition_roles) != 2
					or any(
						isinstance(value, bool) or not isinstance(value, int)
						or value < 0
						for value in age20_roles + reacquisition_roles
					)
					or supervision['successful_updates']
					!= self._belief_aux_updates
					or supervision['successful_updates']
					+ supervision['no_teacher_skips']
					!= supervision['attempts']
					or supervision['age20_available_updates']
					> supervision['attempts']
					or supervision['reacquisition_available_updates']
					> supervision['attempts']
					or min(age20_roles) < 1
					or min(reacquisition_roles) < 1
				):
					raise RuntimeError(
						'Learned-belief checkpoint supervision counts are inconsistent.'
					)
				self._belief_aux_attempts = supervision['attempts']
				self._belief_aux_no_teacher_skips = supervision[
					'no_teacher_skips'
				]
				self._belief_age20_available_updates = supervision[
					'age20_available_updates'
				]
				self._belief_age20_teacher_roles = list(age20_roles)
				self._belief_reacquisition_available_updates = supervision[
					'reacquisition_available_updates'
				]
				self._belief_reacquisition_teacher_roles = list(
					reacquisition_roles
				)
		self.reset_object_belief()
		return

	@torch.no_grad()
	def reset_object_belief(self):
		"""Clear all episode-local learned-belief state."""
		had_state = (
			self._online_object_belief is not None
			or self._online_object_belief_action is not None
		)
		self._online_object_belief = None
		self._online_object_belief_action = None
		if self._learned_object_belief and had_state:
			self._online_object_belief_resets += 1

	@torch.no_grad()
	def set_object_belief_control_for_evaluation(self, enabled):
		"""Switch a frozen learned-belief checkpoint between deployment arms.

		This cannot be used while training or midway through an episode. Shadow
		training runners leave it disabled; held-out evaluators create a fresh
		process and enable it before the first reset when testing the prior.
		"""
		if not self._learned_object_belief:
			raise RuntimeError('Cannot switch control without a learned object belief.')
		if self.training or self.model.training:
			raise RuntimeError('Belief control may only be switched in evaluation mode.')
		if (
			self._online_object_belief is not None
			or self._online_object_belief_action is not None
		):
			raise RuntimeError('Belief control cannot be switched during an episode.')
		if not isinstance(enabled, bool):
			raise TypeError('Belief-control switch must be a bool.')
		self._use_object_belief_for_control = enabled

	@torch.no_grad()
	def _observe_object_belief(self, obs, *, t0):
		"""Apply the same causal role correction used by auxiliary training."""
		if not self._learned_object_belief:
			raise RuntimeError('Learned object belief is disabled.')
		if t0:
			self.reset_object_belief()
		measurement, objects = self.model.encode(
			obs, task=None, return_object=True
		)
		if (
			self._online_object_belief is None
			or self._online_object_belief_action is None
		):
			belief = measurement
		else:
			prior = self.model.belief_prior(
				self._online_object_belief,
				self._online_object_belief_action,
			)
			belief, valid = self.model.belief_posterior(
				prior, measurement, objects
			)
			self._online_object_belief_prior_uses += int(
				(~valid).sum().item()
			)
		self._online_object_belief = belief.detach()
		return belief

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self._use_object_belief_for_control:
			if task is not None or self.cfg.multitask:
				raise RuntimeError(
					'Learned Cutie belief currently supports single-task control only.'
				)
			belief = self._observe_object_belief(obs, t0=bool(t0))
			if self.cfg.mpc:
				action = self.belief_plan(
					belief, t0=t0, eval_mode=eval_mode, task=None
				)
			else:
				action, info = self.model.pi(belief, None)
				if eval_mode:
					action = info['mean']
				action = action[0]
			action = action.clamp(-1, 1)
			# This exact returned action is the one subsequently passed to env.step.
			self._online_object_belief_action = action.detach().reshape(1, -1)
			return action.cpu()
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def act_from_latent(self, z, t0=False, eval_mode=False, task=None):
		"""Plan from a caller-supplied CutieObjectOnly latent state.

		This is a deliberately narrow evaluation-only interface for causal latent
		state diagnostics.  The normal :meth:`act` and compiled planning paths are
		left untouched so training and published checkpoint evaluation cannot
		silently opt into externally maintained state.
		"""
		if not self._cutie_object_only:
			raise RuntimeError(
				'act_from_latent is restricted to structural CutieObjectOnly agents.'
			)
		if self.cfg.compile:
			raise RuntimeError(
				'act_from_latent is evaluation-only and requires compile=false.'
			)
		if self.training or self.model.training:
			raise RuntimeError('act_from_latent requires agent.eval().')
		if not self.cfg.mpc:
			raise RuntimeError('act_from_latent requires MPC planning.')
		if self.cfg.multitask or task is not None:
			raise RuntimeError(
				'act_from_latent currently supports only single-task evaluation.'
			)
		if not isinstance(z, torch.Tensor):
			raise TypeError('act_from_latent expects a torch.Tensor latent state.')
		expected_dim = int(self.cfg.get('cutie_object_only_latent_dim', 128))
		if tuple(z.shape) != (1, expected_dim):
			raise ValueError(
				'act_from_latent expects shape '
				f'(1, {expected_dim}), got {tuple(z.shape)}.'
			)
		if z.device != self.device:
			raise ValueError(
				f'act_from_latent expects {self.device}, got {z.device}.'
			)
		if z.dtype != self._prev_mean.dtype or not torch.isfinite(z).all():
			raise ValueError(
				'act_from_latent requires a finite latent with model dtype '
				f'{self._prev_mean.dtype}, got {z.dtype}.'
			)
		return self._plan_from_latent(
			z, t0=t0, eval_mode=eval_mode, task=None
		).cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	@torch.no_grad()
	def _plan_from_latent(self, z, t0=False, eval_mode=False, task=None):
		"""Evaluation-only MPPI path starting from an already encoded latent.

		The body intentionally mirrors ``_plan`` after its single encode call.
		Keeping it separate avoids changing the established observation planning
		path or its compile/RNG behavior.
		"""
		if task is not None:
			raise RuntimeError('_plan_from_latent supports only task=None.')
		if tuple(z.shape) != (1, int(self.cfg.latent_dim)):
			raise ValueError(
				'_plan_from_latent requires one latent state with shape '
				f'(1, {int(self.cfg.latent_dim)}), got {tuple(z.shape)}.'
			)

		# Sample policy trajectories.
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim,
				device=self.device,
			)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters.
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(
			self.cfg.horizon, self.cfg.action_dim, device=self.device
		)
		std = torch.full(
			(self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std,
			dtype=torch.float, device=self.device,
		)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(
			self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim,
			device=self.device,
		)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI with the exact established random draw order.
		for _ in range(self.cfg.iterations):
			r = torch.randn(
				self.cfg.horizon,
				self.cfg.num_samples-self.cfg.num_pi_trajs,
				self.cfg.action_dim,
				device=std.device,
			)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(
				value.squeeze(1), self.cfg.num_elites, dim=0
			).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (
				score.unsqueeze(0) * elite_actions
			).sum(dim=1) / (score.sum(0) + 1e-9)
			std = (
				(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2)
				.sum(dim=1) / (score.sum(0) + 1e-9)
			).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)

		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		if self._hybrid_graph:
			pi_official_grad_norm = torch.nn.utils.clip_grad_norm_(
				self._official_pi_params, self.cfg.grad_clip_norm
			)
			pi_graph_grad_norm = torch.nn.utils.clip_grad_norm_(
				self._graph_pi_params, self.cfg.grad_clip_norm
			)
			pi_grad_norm = torch.maximum(pi_official_grad_norm, pi_graph_grad_norm)
		else:
			pi_grad_norm = torch.nn.utils.clip_grad_norm_(
				self.model._pi.parameters(), self.cfg.grad_clip_norm
			)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = {
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		}
		if self._hybrid_graph:
			info.update(
				pi_official_grad_norm=pi_official_grad_norm,
				pi_graph_grad_norm=pi_graph_grad_norm,
			)
		return info

	def _sample_object_belief_missing(self, obs):
		"""Draw one isolated contiguous role burst per clean replay sequence."""
		objects = obs['object']
		if objects.ndim != 4 or tuple(objects.shape[-2:]) != (2, 1770):
			raise ValueError(
				'Belief burst sampling expects object observations [T,B,2,1770].'
			)
		time, batch = objects.shape[:2]
		burn_in = int(self.cfg.get('cutie_object_belief_burn_in', 3))
		minimum = int(self.cfg.get('cutie_object_belief_min_burst', 5))
		maximum = int(self.cfg.get('cutie_object_belief_max_burst', 20))
		if not (1 <= minimum <= maximum):
			raise ValueError('Belief burst bounds must satisfy 1 <= min <= max.')
		if burn_in < 1 or burn_in + maximum > time:
			raise ValueError(
				f'Belief sequence length {time} cannot hold burn-in {burn_in} '
				f'and maximum burst {maximum}.'
			)
		length = torch.randint(
			minimum, maximum + 1, (batch,),
			generator=self._belief_mask_generator,
		)
		role = torch.randint(
			0, cutie_object_belief.NUM_ROLES, (batch,),
			generator=self._belief_mask_generator,
		)
		# Guarantee age-max supervision for both roles. Remaining sequences keep
		# the configured uniform role and burst-length distribution.
		if batch >= cutie_object_belief.NUM_ROLES:
			length[:cutie_object_belief.NUM_ROLES] = maximum
			role[:cutie_object_belief.NUM_ROLES] = torch.arange(
				cutie_object_belief.NUM_ROLES, dtype=role.dtype
			)
		steps = torch.arange(time).view(time, 1)
		active = (steps >= burn_in) & (steps < burn_in + length.view(1, batch))
		role_mask = torch.nn.functional.one_hot(
			role, num_classes=cutie_object_belief.NUM_ROLES
		).to(torch.bool)
		missing = active.unsqueeze(-1) & role_mask.unsqueeze(0)
		if bool(missing[0].any().item()):
			raise RuntimeError('Belief mask generator corrupted the causal first frame.')
		return missing.to(device=objects.device, non_blocking=True)

	def _update_object_belief(self, obs, action, missing_latest):
		"""Train only the dedicated belief transition on a clean-teacher rollout."""
		if not self._learned_object_belief or self.belief_optim is None:
			raise RuntimeError('Object belief auxiliary update is disabled.')
		objects = obs['object']
		expected_time = (
			int(self.cfg.get('cutie_object_belief_burn_in', 3))
			+ int(self.cfg.get('cutie_object_belief_max_burst', 20))
			+ int(self.cfg.get('cutie_object_belief_recovery_frames', 1))
		)
		if objects.shape[0] != expected_time or action.shape[0] != expected_time - 1:
			raise ValueError(
				'Belief auxiliary sequence/action length mismatch: '
				f'{objects.shape[0]}, {action.shape[0]}, expected {expected_time}.'
			)
		if missing_latest.shape != objects.shape[:3] or missing_latest.dtype != torch.bool:
			raise ValueError('Belief auxiliary missing mask has an invalid schema.')
		self._belief_aux_attempts += 1

		# Replay stays immutable and supplies a detached clean teacher. The policy
		# measurement is rebuilt from raw missing frames, including their overlap
		# in the next two three-frame stacks.
		with torch.no_grad():
			clean_measurement, clean_objects = self.model.encode(
				obs, task=None, return_object=True
			)
			corrupted_objects, stack_affected = (
				cutie_object_belief.rebuild_stacks_with_missing_frames(
					clean_objects, missing_latest
				)
			)
			corrupted_measurement = self.model._encoder['object'](
				corrupted_objects
			)
			clean_valid = cutie_object_belief.latest_valid(clean_objects)

		belief = corrupted_measurement[0]
		role_losses = []
		role_counts = []
		burn_in = int(self.cfg.get('cutie_object_belief_burn_in', 3))
		maximum = int(self.cfg.get('cutie_object_belief_max_burst', 20))
		age_loss = {}
		age_count = {}
		age_role_count = {}
		reacquisition_numerator = objects.new_zeros(())
		reacquisition_count = objects.new_zeros(())
		reacquisition_role_count = objects.new_zeros(
			(cutie_object_belief.NUM_ROLES,)
		)
		pending_reacquisition = torch.zeros(
			objects.shape[1], cutie_object_belief.NUM_ROLES,
			dtype=torch.bool, device=objects.device,
		)
		for t in range(1, expected_time):
			prior = self.model.belief_prior(belief, action[t - 1])
			prior_roles = prior.reshape(
				prior.shape[0], cutie_object_belief.NUM_ROLES,
				cutie_object_belief.ROLE_DIM,
			)
			target_roles = clean_measurement[t].reshape_as(prior_roles)
			prior_per_role = (prior_roles - target_roles).square().mean(dim=-1)
			pending_reacquisition, reacquired = (
				cutie_object_belief.advance_reacquisition_pending(
					pending_reacquisition,
					missing_latest[t - 1],
					missing_latest[t],
					clean_valid[t],
				)
			)
			reacquisition_numerator = (
				reacquisition_numerator
				+ (prior_per_role * reacquired.to(prior_per_role.dtype)).sum()
			)
			reacquisition_count = reacquisition_count + reacquired.sum()
			reacquisition_role_count = (
				reacquisition_role_count
				+ reacquired.to(reacquisition_role_count.dtype).sum(dim=0)
			)
			belief, _ = self.model.belief_posterior(
				prior, corrupted_measurement[t], corrupted_objects[t]
			)
			if burn_in <= t < burn_in + maximum:
				belief_roles = belief.reshape(
					belief.shape[0], cutie_object_belief.NUM_ROLES,
					cutie_object_belief.ROLE_DIM,
				)
				target_roles = clean_measurement[t].reshape_as(belief_roles)
				per_role = (belief_roles - target_roles).square().mean(dim=-1)
				weight = (missing_latest[t] & clean_valid[t]).to(per_role.dtype)
				count = weight.sum()
				loss = (per_role * weight).sum() / count.clamp_min(1.0)
				role_losses.append(loss)
				role_counts.append(count)
				age = t - burn_in + 1
				age_loss[age] = loss
				age_count[age] = count
				age_role_count[age] = weight.sum(dim=0)
		if len(role_losses) != maximum:
			raise RuntimeError('Belief auxiliary did not cover every configured age.')
		losses = torch.stack(role_losses)
		counts = torch.stack(role_counts)
		missing_loss, available_ages = (
			cutie_object_belief.mean_over_available_teacher_groups(
				losses, counts
			)
		)
		reacquisition_loss = (
			reacquisition_numerator
			/ reacquisition_count.to(objects.dtype).clamp_min(1.0)
		)
		age20_roles = [
			int(value)
			for value in age_role_count[maximum].detach().to('cpu').tolist()
		]
		reacquisition_roles = [
			int(value)
			for value in reacquisition_role_count.detach().to('cpu').tolist()
		]
		age20_available = sum(age20_roles) > 0
		reacquisition_available = sum(reacquisition_roles) > 0
		self._belief_age20_available_updates += int(age20_available)
		self._belief_reacquisition_available_updates += int(
			reacquisition_available
		)
		for role_index in range(cutie_object_belief.NUM_ROLES):
			self._belief_age20_teacher_roles[role_index] += age20_roles[
				role_index
			]
			self._belief_reacquisition_teacher_roles[role_index] += (
				reacquisition_roles[role_index]
			)
		any_teacher = bool(available_ages.any().item()) or reacquisition_available
		if not any_teacher:
			self._belief_aux_no_teacher_skips += 1
			# Even an all-zero gradient Adam step can change parameters through
			# optimizer momentum.  A label-free batch must therefore skip the
			# optimizer entirely and clear any stale gradients defensively.
			self.belief_optim.zero_grad(set_to_none=True)
			zero = objects.new_zeros(())
			return {
				'belief_loss': zero,
				'belief_missing_loss': zero,
				'belief_reacquisition_loss': zero,
				'belief_reacquisition_teacher_roles': reacquisition_count.detach(),
				'belief_weighted_loss': zero,
				'belief_grad_norm': zero,
				'belief_teacher_role_steps': counts.sum().detach(),
				'belief_supervised_age_count': available_ages.sum().detach(),
				'belief_age20_loss': age_loss[maximum].detach(),
				'belief_age20_teacher_roles': age_count[maximum].detach(),
				'belief_age20_supervision_available': zero,
				'belief_reacquisition_supervision_available': zero,
				'belief_pending_reacquisitions': pending_reacquisition.sum().detach(),
				'belief_synthetic_role_frames': missing_latest.sum().detach(),
				'belief_affected_stack_roles': stack_affected.sum().detach(),
				'belief_aux_attempt': zero.new_ones(()),
				'belief_aux_no_teacher_skip': zero.new_ones(()),
				'belief_aux_update': zero,
			}
		belief_loss = (
			missing_loss
			+ float(self.cfg.get(
				'cutie_object_belief_reacquisition_coef', 1.0
			)) * reacquisition_loss
		)
		weighted_loss = float(self.cfg.get(
			'cutie_object_belief_loss_coef', 1.0
		)) * belief_loss
		self.belief_optim.zero_grad(set_to_none=True)
		weighted_loss.backward()
		belief_grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model._belief_dynamics.parameters(), self.cfg.grad_clip_norm
		)
		self.belief_optim.step()
		self._belief_aux_updates += 1
		self.belief_optim.zero_grad(set_to_none=True)
		self.model.eval()
		return {
			'belief_loss': belief_loss.detach(),
			'belief_missing_loss': missing_loss.detach(),
			'belief_reacquisition_loss': reacquisition_loss.detach(),
			'belief_reacquisition_teacher_roles': reacquisition_count.detach(),
			'belief_weighted_loss': weighted_loss.detach(),
			'belief_grad_norm': belief_grad_norm.detach(),
			'belief_teacher_role_steps': counts.sum().detach(),
			'belief_supervised_age_count': available_ages.sum().detach(),
			'belief_age20_loss': age_loss[maximum].detach(),
			'belief_age20_teacher_roles': age_count[maximum].detach(),
			'belief_age20_supervision_available': belief_loss.new_tensor(
				float(age20_available)
			),
			'belief_reacquisition_supervision_available': belief_loss.new_tensor(
				float(reacquisition_available)
			),
			'belief_pending_reacquisitions': pending_reacquisition.sum().detach(),
			'belief_synthetic_role_frames': missing_latest.sum().detach(),
			'belief_affected_stack_roles': stack_affected.sum().detach(),
			'belief_aux_attempt': belief_loss.new_ones(()),
			'belief_aux_no_teacher_skip': belief_loss.new_zeros(()),
			'belief_aux_update': belief_loss.new_ones(()),
		}

	def _sample_real_sibling_shift(self, batch_size):
		if self._rof_real_sibling_shift_generator is None:
			raise RuntimeError('Real-sibling augmentation RNG is unavailable.')
		encoder = self.model._encoder['object']
		pad = int(encoder.augmentation.pad)
		if pad == 0:
			return torch.zeros(
				batch_size, 1, 1, 2, device=self.device, dtype=torch.float32
			)
		return torch.randint(
			0, 2 * pad + 1, (batch_size, 1, 1, 2),
			device=self.device, dtype=torch.float32,
			generator=self._rof_real_sibling_shift_generator,
		)

	def _real_sibling_objective(
		self, root, positive_future, negative_future, action,
	):
		"""Build the auxiliary term for the *same* base optimizer step."""
		if not self._rof_real_sibling_aux_enabled:
			raise RuntimeError('Real-sibling auxiliary objective is disabled.')
		encoder = self.model._encoder['object']
		root_shift = self._sample_real_sibling_shift(action.shape[1])
		z = encoder(root, shift_index=root_shift)
		predicted = []
		positive_targets = []
		negative_targets = []
		for index, step_action in enumerate(action.unbind(0)):
			z = self.model.next(z, step_action, None)
			predicted.append(z)
			# The positive and different-action real future share exactly the same
			# crop.  Ranking therefore cannot exploit augmentation noise.
			shift = self._sample_real_sibling_shift(step_action.shape[0])
			with torch.no_grad():
				positive_targets.append(encoder({
					field: positive_future[field][index]
					for field in rof_real_sibling_auxiliary.POLICY_FIELDS
				}, shift_index=shift))
				negative_targets.append(encoder({
					field: negative_future[field][index]
					for field in rof_real_sibling_auxiliary.POLICY_FIELDS
				}, shift_index=shift))
		return rof_real_sibling_auxiliary.latent_objective(
			torch.stack(predicted),
			torch.stack(positive_targets),
			torch.stack(negative_targets),
			rho=float(self.cfg.rho),
			positive_coef=float(self.cfg.rof_real_sibling_aux_positive_coef),
			ranking_coef=float(self.cfg.rof_real_sibling_aux_ranking_coef),
			margin_fraction=float(self.cfg.rof_real_sibling_aux_margin_fraction),
		)

	def _sample_rgb_interventional_shift(self, batch_size):
		"""Draw auxiliary-owned crop indices without touching TD-MPC2's RNG."""
		if self._rgb_interventional_shift_generator is None:
			raise RuntimeError('RGB interventional augmentation RNG is unavailable.')
		modules = list(self.model._encoder['rgb'].children())
		if not modules or modules[0].__class__.__name__ != 'ShiftAug':
			raise RuntimeError('Expected the official RGB encoder to start with ShiftAug.')
		pad = int(modules[0].pad)
		self._rgb_interventional_shift_shape_trace.update(
			f'{self._rgb_interventional_shift_calls}:{int(batch_size)};'.encode(
				'ascii'
			)
		)
		self._rgb_interventional_shift_calls += 1
		self._rgb_interventional_shift_rows += int(batch_size)
		if pad == 0:
			return torch.zeros(
				batch_size, 1, 1, 2, device=self.device, dtype=torch.int64
			)
		return torch.randint(
			0, 2 * pad + 1, (batch_size, 1, 1, 2),
			device=self.device, dtype=torch.int64,
			generator=self._rgb_interventional_shift_generator,
		)

	def _encode_rgb_interventional_shared(self, rgb, shift_index):
		"""Run the official RGB encoder with an explicit, shareable ShiftAug crop."""
		if rgb.ndim != 4 or tuple(rgb.shape[-2:]) != (64, 64):
			raise ValueError('RGB intervention expects [B,9,64,64].')
		if shift_index.shape != (rgb.shape[0], 1, 1, 2):
			raise ValueError('RGB intervention crop shape does not match its batch.')
		modules = list(self.model._encoder['rgb'].children())
		if not modules or modules[0].__class__.__name__ != 'ShiftAug':
			raise RuntimeError('RGB intervention requires the official RGB encoder.')
		pad = int(modules[0].pad)
		x = rgb.float()
		if pad:
			batch, _, height, width = x.shape
			x = F.pad(x, (pad,) * 4, mode='replicate')
			padded = height + 2 * pad
			eps = 1.0 / padded
			axis = torch.linspace(
				-1.0 + eps, 1.0 - eps, padded,
				device=x.device, dtype=x.dtype,
			)[:height]
			axis = axis.unsqueeze(0).repeat(height, 1).unsqueeze(2)
			base_grid = torch.cat([axis, axis.transpose(1, 0)], dim=2)
			base_grid = base_grid.unsqueeze(0).repeat(batch, 1, 1, 1)
			grid = base_grid + shift_index.to(x.dtype) * (2.0 / padded)
			x = F.grid_sample(
				x, grid, padding_mode='zeros', align_corners=False
			)
		for module in modules[1:]:
			x = module(x)
		return x

	def _rgb_interventional_forward(
		self, clean_root, hard_root,
		clean_positive_future, hard_positive_future,
		clean_negative_future, hard_negative_future,
		action,
	):
		"""Shared forward path for joint and strict data-matched arms.

		Both arms call this exact function once per optimizer update.  Consequently
		they encode the same root/future tensors in the same call order, consume the
		same augmentation draws with the same shapes, and roll out the same sampled
		action tensor.  Only the downstream loss is allowed to differ.
		"""
		if not self._rgb_interventional_aux_enabled:
			raise RuntimeError('RGB interventional forward path is disabled.')
		if action.ndim != 3:
			raise ValueError('RGB interventional action must have shape [H,B,A].')
		batch_size = action.shape[1]
		root_shift = self._sample_rgb_interventional_shift(batch_size)
		clean_root_z = self._encode_rgb_interventional_shared(
			clean_root, root_shift
		)
		hard_root_z = self._encode_rgb_interventional_shared(
			hard_root, root_shift
		)

		clean_positive_z, hard_positive_z = [], []
		clean_negative_z, hard_negative_z = [], []
		for index in range(action.shape[0]):
			# All four corresponding renders use the same explicit crop. Neither
			# background identity nor augmentation noise can solve the fork ranking.
			shift = self._sample_rgb_interventional_shift(batch_size)
			clean_positive_z.append(self._encode_rgb_interventional_shared(
				clean_positive_future[index], shift
			))
			hard_positive_z.append(self._encode_rgb_interventional_shared(
				hard_positive_future[index], shift
			))
			clean_negative_z.append(self._encode_rgb_interventional_shared(
				clean_negative_future[index], shift
			))
			hard_negative_z.append(self._encode_rgb_interventional_shared(
				hard_negative_future[index], shift
			))
		clean_positive_z = torch.stack(clean_positive_z)
		hard_positive_z = torch.stack(hard_positive_z)
		clean_negative_z = torch.stack(clean_negative_z)
		hard_negative_z = torch.stack(hard_negative_z)
		clean_prediction, hard_prediction = [], []
		clean_z, hard_z = clean_root_z, hard_root_z
		for step_action in action.unbind(0):
			clean_z = self.model.next(clean_z, step_action, None)
			hard_z = self.model.next(hard_z, step_action, None)
			clean_prediction.append(clean_z)
			hard_prediction.append(hard_z)
		return {
			'clean_root': clean_root_z,
			'hard_root': hard_root_z,
			'clean_positive': clean_positive_z,
			'hard_positive': hard_positive_z,
			'clean_negative': clean_negative_z,
			'hard_negative': hard_negative_z,
			'clean_prediction': torch.stack(clean_prediction),
			'hard_prediction': torch.stack(hard_prediction),
		}

	def _rgb_interventional_objective(
		self, clean_root, hard_root,
		clean_positive_future, hard_positive_future,
		clean_negative_future, hard_negative_future,
		action, outcome_gap, eligible,
		background_pairing=None, fork_pairing=None,
	):
		"""Joint objective with correct or auditable shuffled correspondences."""
		forward = self._rgb_interventional_forward(
			clean_root, hard_root,
			clean_positive_future, hard_positive_future,
			clean_negative_future, hard_negative_future,
			action,
		)
		batch_size = action.shape[1]
		identity = torch.arange(batch_size, device=action.device)
		if background_pairing is None:
			background_pairing = identity
		if fork_pairing is None:
			fork_pairing = identity
		for name, indices in (
			('background', background_pairing), ('fork', fork_pairing),
		):
			if (
				indices.dtype != torch.int64
				or indices.device != action.device
				or tuple(indices.shape) != (batch_size,)
				or not torch.equal(torch.sort(indices).values, identity)
			):
				raise ValueError(f'RGB {name} pairing must be a device-local permutation.')
		hard_background = torch.cat([
			forward['hard_root'].unsqueeze(0),
			forward['hard_positive'], forward['hard_negative'],
		], dim=0).index_select(1, background_pairing)
		background_loss = rgb_interventional_auxiliary.background_invariance_loss(
			torch.cat([
				forward['clean_root'].unsqueeze(0),
				forward['clean_positive'], forward['clean_negative'],
			], dim=0),
			hard_background,
		)
		clean_negative = forward['clean_negative'].index_select(1, fork_pairing)
		hard_negative = forward['hard_negative'].index_select(1, fork_pairing)
		shuffled_gap = outcome_gap.index_select(1, fork_pairing)
		shuffled_eligible = eligible.index_select(1, fork_pairing)
		fork = rgb_interventional_auxiliary.outcome_grounded_fork_objective(
			torch.cat([
				forward['clean_prediction'], forward['hard_prediction'],
			], dim=1),
			torch.cat([
				forward['clean_positive'], forward['hard_positive'],
			], dim=1),
			torch.cat([
				clean_negative, hard_negative,
			], dim=1),
			shuffled_gap.repeat(1, 2),
			shuffled_eligible.repeat(1, 2),
			rho=float(self.cfg.rho),
			positive_coef=float(self.cfg.rgb_interventional_aux_positive_coef),
			separation_coef=float(self.cfg.rgb_interventional_aux_separation_coef),
			ranking_coef=float(self.cfg.rgb_interventional_aux_ranking_coef),
			margin_min=float(self.cfg.rgb_interventional_aux_margin_min),
			margin_max=float(self.cfg.rgb_interventional_aux_margin_max),
		)
		background_weighted = (
			float(self.cfg.rgb_interventional_aux_background_coef)
			* background_loss
		)
		return {
			**fork,
			'background_loss': background_loss,
			'background_weighted_loss': background_weighted,
			'weighted_loss': background_weighted + fork['weighted_loss'],
		}

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(
		self, obs, action, reward, terminated, task=None, state_targets=None,
		sibling_root=None, sibling_positive_future=None,
		sibling_negative_future=None, sibling_action=None,
		rgb_aux_clean_root=None, rgb_aux_hard_root=None,
		rgb_aux_clean_positive_future=None, rgb_aux_hard_positive_future=None,
		rgb_aux_clean_negative_future=None, rgb_aux_hard_negative_future=None,
		rgb_aux_action=None, rgb_aux_negative_action=None,
		rgb_aux_outcome_gap=None, rgb_aux_eligible=None,
		rgb_aux_background_pairing=None, rgb_aux_fork_pairing=None,
	):
		sibling_values = (
			sibling_root, sibling_positive_future,
			sibling_negative_future, sibling_action,
		)
		if self._rof_real_sibling_aux_enabled != all(
			value is not None for value in sibling_values
		):
			raise RuntimeError(
				'Real-sibling tensors must be present exactly when its auxiliary is enabled.'
			)
		rgb_aux_transition_values = (
			rgb_aux_clean_root, rgb_aux_hard_root,
			rgb_aux_clean_positive_future, rgb_aux_hard_positive_future,
			rgb_aux_clean_negative_future, rgb_aux_hard_negative_future,
			rgb_aux_action, rgb_aux_negative_action,
		)
		if self._rgb_interventional_aux_enabled != all(
			value is not None for value in rgb_aux_transition_values
		):
			raise RuntimeError(
				'RGB auxiliary transition tensors must be present exactly when enabled.'
			)
		if (
			self._rgb_interventional_aux_enabled
			and rgb_aux_action.shape != rgb_aux_negative_action.shape
		):
			raise RuntimeError(
				'RGB sampled positive/negative action tensors must have equal shape.'
			)
		rgb_label_values = (rgb_aux_outcome_gap, rgb_aux_eligible)
		rgb_data_matched = bool(
			self._rgb_interventional_aux_enabled
			and self._rgb_interventional_aux_contract is not None
			and self._rgb_interventional_aux_contract.get('arm') == 'data_matched'
		)
		labels_required = self._rgb_interventional_aux_enabled
		if labels_required != all(value is not None for value in rgb_label_values):
			raise RuntimeError(
				'RGB pair targets must be present only for intervention-specific arms.'
			)
		if False and any(value is not None for value in rgb_label_values):
			raise RuntimeError('RGB data-matched updates forbid pair targets.')
		pairing_values = (rgb_aux_background_pairing, rgb_aux_fork_pairing)
		if self._rgb_interventional_aux_enabled != all(
			value is not None for value in pairing_values
		):
			raise RuntimeError(
				'RGB pairing indices must be present exactly when the auxiliary is enabled.'
			)
		anchor_mode = self.cfg.get('flat_anchor_mode', 'residual')
		structured_anchor = (
			self.cfg.get('flat_anchor', False)
			and anchor_mode in {
				'oc', 'object_graph', 'hybrid_graph', 'reward_graph',
				'cutie_hybrid', 'cutie_object_only',
			}
		)
		# Legacy OC intentionally shares one crop across its sequence. HybridGraph
		# instead samples inside each encode call so RGB follows the official
		# per-frame ShiftAug RNG/order while each anchor shares its own RGB shift.
		oc_shift = self.model.sample_oc_shift(
			obs['anchor'].shape[1], obs['anchor'].device
		) if structured_anchor and anchor_mode == 'oc' else None
		# Compute targets
		with torch.no_grad():
			if structured_anchor:
				if self._cutie_object_mode:
					next_z, next_anchor = self.model.encode(
						obs[1:], task, return_object=True
					)
				else:
					next_z, next_anchor = self.model.encode(
						obs[1:], task, return_anchor=True, oc_shift=oc_shift
					)
			else:
				next_z = self.model.encode(obs[1:], task)
				next_anchor = None
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		if structured_anchor:
			if self._cutie_object_mode:
				z, current_anchor = self.model.encode(
					obs[0], task, return_object=True
				)
			else:
				z, current_anchor = self.model.encode(
					obs[0], task, return_anchor=True, oc_shift=oc_shift
				)
		else:
			z = self.model.encode(obs[0], task)
			current_anchor = None
		zs[0] = z
		consistency_loss = 0
		rgb_consistency_loss = 0
		graph_consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			if self._hybrid_graph or self._reward_graph:
				scene_dim = self.cfg.flat_anchor_scene_dim
				rgb_consistency_loss = rgb_consistency_loss + F.mse_loss(
					z[..., :scene_dim], _next_z[..., :scene_dim]
				) * self.cfg.rho**t
				graph_consistency_loss = graph_consistency_loss + F.mse_loss(
					z[..., scene_dim:], _next_z[..., scene_dim:]
				) * self.cfg.rho**t
			else:
				consistency_loss = consistency_loss + F.mse_loss(
					z, _next_z
				) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		if self._hybrid_graph or self._reward_graph:
			rgb_consistency_loss = rgb_consistency_loss / self.cfg.horizon
			graph_consistency_loss = graph_consistency_loss / self.cfg.horizon
			consistency_loss = rgb_consistency_loss + graph_consistency_loss
		else:
			consistency_loss = consistency_loss / self.cfg.horizon
			if object_state_supervision.bottleneck_enabled(self.cfg):
				# MSE normally averages over 128 latent coordinates. Only state_dim
				# coordinates carry information in this strict bottleneck.
				consistency_loss = consistency_loss * (
					float(self.cfg.latent_dim)
					/ float(object_state_supervision.target_dim(self.cfg))
				)
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			# Keep every compiled output on the model device. A Python zero becomes
			# a CPU tensor in the metrics conversion and disables CUDA graphs.
			termination_loss = zs.new_zeros(())
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		if structured_anchor:
			if self._cutie_object_mode:
				anchor_reconstruction_loss = self.model.object_loss(
					self.model.decode_object(zs[0]),
					current_anchor,
				)
				anchor_predictions = self.model.decode_object(zs[1:])
				loss_fn = self.model.object_loss
			else:
				anchor_reconstruction_loss = self.model.anchor_loss(
					self.model.decode_anchor(zs[0]),
					current_anchor,
				)
				anchor_predictions = self.model.decode_anchor(zs[1:])
				loss_fn = self.model.anchor_loss
			anchor_prediction_loss = 0
			for t, (anchor_prediction, anchor_target) in enumerate(zip(
				anchor_predictions.unbind(0), next_anchor.unbind(0)
			)):
				anchor_prediction_loss = anchor_prediction_loss + loss_fn(
					anchor_prediction,
					anchor_target,
				) * self.cfg.rho**t
			anchor_prediction_loss = anchor_prediction_loss / self.cfg.horizon
		else:
			anchor_reconstruction_loss = 0.
			anchor_prediction_loss = 0.
		consistency_term = (
			self.cfg.consistency_coef * rgb_consistency_loss
			+ self.cfg.flat_anchor_graph_consistency_coef * graph_consistency_loss
			if self._hybrid_graph or self._reward_graph
			else self.cfg.consistency_coef * consistency_loss
		)
		total_loss = (
			consistency_term +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)
		if structured_anchor:
			anchor_reconstruction_coef = float(
				self.cfg.flat_anchor_reconstruction_coef
			)
			anchor_prediction_coef = float(
				self.cfg.flat_anchor_prediction_coef
			)
			if self._robust_object_field_auxiliary_contract is not None:
				anchor_reconstruction_coef = self._robust_object_field_auxiliary_contract[
					'effective_reconstruction_coef'
				]
				anchor_prediction_coef = self._robust_object_field_auxiliary_contract[
					'effective_prediction_coef'
				]
			total_loss = (
				total_loss
				+ anchor_reconstruction_coef * anchor_reconstruction_loss
				+ anchor_prediction_coef * anchor_prediction_loss
			)

		if object_state_supervision.enabled(self.cfg):
			# Targets are a separate replay field, aligned with obs[0:H+1].
			# The initial estimate and action-conditioned rollout receive the
			# same normalized state target schema. No teacher forcing is used.
			state_predictions = self.model.decode_supervised_state(zs)
			state_frame_errors = (state_predictions - state_targets.detach()).square().mean(-1)
			state_reconstruction_loss = state_frame_errors[0].mean()
			state_prediction_loss = sum(
				state_frame_errors[t + 1].mean() * self.cfg.rho**t
				for t in range(self.cfg.horizon)
			) / self.cfg.horizon
			state_supervision_loss = 0.5 * (state_reconstruction_loss + state_prediction_loss)
			state_supervision_weighted = float(self.cfg.object_state_supervision_coef) * state_supervision_loss
			total_loss = total_loss + state_supervision_weighted

		# B and C add their real-sibling term to the established objective before
		# the one and only backward/clip/Adam step.  There is no second optimizer
		# update, so A/B/C retain the same number of optimizer steps.
		sibling_objective = None
		if self._rof_real_sibling_aux_enabled:
			sibling_objective = self._real_sibling_objective(
				sibling_root, sibling_positive_future,
				sibling_negative_future, sibling_action,
			)
			total_loss = total_loss + sibling_objective['weighted_loss']

		rgb_interventional_objective = None
		if self._rgb_interventional_aux_enabled:
			if False:
				rgb_interventional_objective = self._rgb_data_matched_objective(
					rgb_aux_clean_root, rgb_aux_hard_root,
					rgb_aux_clean_positive_future, rgb_aux_hard_positive_future,
					rgb_aux_clean_negative_future, rgb_aux_hard_negative_future,
					rgb_aux_action,
				)
			else:
				rgb_interventional_objective = self._rgb_interventional_objective(
					rgb_aux_clean_root, rgb_aux_hard_root,
					rgb_aux_clean_positive_future, rgb_aux_hard_positive_future,
					rgb_aux_clean_negative_future, rgb_aux_hard_negative_future,
					rgb_aux_action, rgb_aux_outcome_gap, rgb_aux_eligible,
					rgb_aux_background_pairing, rgb_aux_fork_pairing,
				)
			total_loss = (
				total_loss + rgb_interventional_objective['weighted_loss']
			)

		# Update model
		total_loss.backward()
		if self._hybrid_graph:
			official_grad_norm = torch.nn.utils.clip_grad_norm_(
				self._official_world_params, self.cfg.grad_clip_norm
			)
			graph_grad_norm = torch.nn.utils.clip_grad_norm_(
				self._graph_world_params, self.cfg.grad_clip_norm
			)
			grad_norm = torch.maximum(official_grad_norm, graph_grad_norm)
		elif self._reward_graph:
			official_grad_norm = torch.nn.utils.clip_grad_norm_(
				self._official_world_params, self.cfg.grad_clip_norm
			)
			graph_aux_grad_norm = torch.nn.utils.clip_grad_norm_(
				self._graph_aux_params, self.cfg.grad_clip_norm
			)
			graph_reward_grad_norm = torch.nn.utils.clip_grad_norm_(
				self._graph_reward_params, self.cfg.grad_clip_norm
			)
			grad_norm = torch.maximum(
				official_grad_norm,
				torch.maximum(graph_aux_grad_norm, graph_reward_grad_norm),
			)
		else:
			grad_norm = torch.nn.utils.clip_grad_norm_(
				self.model.parameters(), self.cfg.grad_clip_norm
			)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = {
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		}
		if object_state_supervision.enabled(self.cfg):
			info.update(
				state_supervision_loss=state_supervision_loss,
				state_reconstruction_loss=state_reconstruction_loss,
				state_prediction_loss=state_prediction_loss,
				state_supervision_weighted=state_supervision_weighted,
			)
		if self._hybrid_graph:
			info.update(
				rgb_consistency_loss=rgb_consistency_loss,
				graph_consistency_loss=graph_consistency_loss,
				official_grad_norm=official_grad_norm,
				graph_grad_norm=graph_grad_norm,
			)
		elif self._reward_graph:
			info.update(
				rgb_consistency_loss=rgb_consistency_loss,
				graph_consistency_loss=graph_consistency_loss,
				official_grad_norm=official_grad_norm,
				graph_aux_grad_norm=graph_aux_grad_norm,
				graph_reward_grad_norm=graph_reward_grad_norm,
			)
		if structured_anchor:
			if self._cutie_object_mode:
				info.update(
					object_reconstruction_loss=anchor_reconstruction_loss,
					object_prediction_loss=anchor_prediction_loss,
				)
				if self._robust_object_field_auxiliary_contract is not None:
					info.update(
						object_reconstruction_coef=anchor_reconstruction_coef,
						object_prediction_coef=anchor_prediction_coef,
						object_reconstruction_weighted_loss=(
							anchor_reconstruction_coef * anchor_reconstruction_loss
						),
						object_prediction_weighted_loss=(
							anchor_prediction_coef * anchor_prediction_loss
						),
					)
			else:
				info.update(
					anchor_reconstruction_loss=anchor_reconstruction_loss,
					anchor_prediction_loss=anchor_prediction_loss,
				)
		if sibling_objective is not None:
			info.update(
				sibling_positive_loss=sibling_objective['positive_loss'],
				sibling_ranking_loss=sibling_objective['ranking_loss'],
				sibling_weighted_loss=sibling_objective['weighted_loss'],
				sibling_positive_mse=sibling_objective['positive_mse'],
				sibling_wrong_mse=sibling_objective['wrong_sibling_mse'],
				sibling_target_distance=sibling_objective[
					'sibling_target_distance'
				],
				sibling_correct_minus_wrong_mse=sibling_objective[
					'correct_minus_wrong_mse'
				],
				sibling_ranking_accuracy=sibling_objective[
					'ranking_accuracy'
				],
				sibling_margin_violation_rate=sibling_objective[
					'margin_violation_rate'
				],
				sibling_joint_grad_norm=grad_norm,
				sibling_aux_attempt=total_loss.new_ones(()),
				sibling_aux_update=total_loss.new_ones(()),
			)
		if rgb_interventional_objective is not None:
			info.update(
				rgb_aux_background_loss=rgb_interventional_objective[
					'background_loss'
				],
				rgb_aux_background_weighted_loss=rgb_interventional_objective[
					'background_weighted_loss'
				],
				rgb_aux_positive_loss=rgb_interventional_objective[
					'positive_loss'
				],
				rgb_aux_separation_loss=rgb_interventional_objective[
					'separation_loss'
				],
				rgb_aux_ranking_loss=rgb_interventional_objective[
					'ranking_loss'
				],
				rgb_aux_weighted_loss=rgb_interventional_objective[
					'weighted_loss'
				],
				rgb_aux_positive_mse=rgb_interventional_objective[
					'positive_mse'
				],
				rgb_aux_wrong_sibling_mse=rgb_interventional_objective[
					'wrong_sibling_mse'
				],
				rgb_aux_target_distance=rgb_interventional_objective[
					'target_distance'
				],
				rgb_aux_eligible_rate=rgb_interventional_objective[
					'eligible_rate'
				],
				rgb_aux_eligible_outcome_gap=rgb_interventional_objective[
					'eligible_outcome_gap'
				],
				rgb_aux_ranking_accuracy=rgb_interventional_objective[
					'ranking_accuracy'
				],
				rgb_aux_margin_violation_rate=rgb_interventional_objective[
					'margin_violation_rate'
				],
				rgb_aux_joint_grad_norm=grad_norm,
				rgb_aux_attempt=total_loss.new_ones(()),
				rgb_aux_update=total_loss.new_ones(()),
			)
			if False:
				info.update(
					rgb_aux_data_matched_consistency_loss=(
						rgb_interventional_objective[
							'data_matched_consistency_loss'
						]
					),
					rgb_aux_pair_targets_used=(
						rgb_interventional_objective['pair_targets_used']
					),
				)
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return {k: v.detach().mean().to(self.device) if isinstance(v, torch.Tensor) \
			else torch.as_tensor(v, device=self.device) \
			for k, v in info.items()}

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		batch = buffer.sample()
		obs, action, reward, terminated, task = batch[:5]
		kwargs = {}
		if object_state_supervision.enabled(self.cfg):
			if len(batch) != 6:
				raise RuntimeError('State-supervised update requires separately stored replay labels.')
			state_targets = batch[5]
			expected = (self.cfg.horizon + 1, self.cfg.batch_size, object_state_supervision.target_dim(self.cfg))
			if tuple(state_targets.shape) != expected or not torch.isfinite(state_targets).all():
				raise RuntimeError('State-supervision replay labels have an invalid shape or nonfinite values.')
			kwargs['state_targets'] = state_targets
		if task is not None:
			kwargs["task"] = task
		sibling_batch = None
		if self._rof_real_sibling_aux_enabled:
			if not self._rof_real_sibling_aux_training:
				raise RuntimeError(
					'Policy-only evaluation cannot execute a training update.'
				)
			if self._rof_real_sibling_aux_replay is None:
				self._rof_real_sibling_aux_replay = (
					rof_real_sibling_auxiliary.Replay(
						self.cfg, device=self.device
					)
				)
			self._rof_real_sibling_aux_attempts += 1
			sibling_batch = self._rof_real_sibling_aux_replay.sample()
			kwargs.update(
				sibling_root=sibling_batch.root,
				sibling_positive_future=sibling_batch.positive_future,
				sibling_negative_future=sibling_batch.negative_future,
				sibling_action=sibling_batch.action,
			)
		rgb_interventional_batch = None
		rgb_pairing_indices = None
		if self._rgb_interventional_aux_enabled:
			if not self._rgb_interventional_aux_training:
				raise RuntimeError(
					'Policy-only evaluation cannot execute RGB auxiliary training.'
				)
			if self._rgb_interventional_aux_replay is None:
				self._rgb_interventional_aux_replay = (
					rgb_interventional_auxiliary.Replay(
						self.cfg, device=self.device
					)
				)
			self._rgb_interventional_aux_attempts += 1
			rgb_interventional_batch = self._rgb_interventional_aux_replay.sample()
			if self._rgb_interventional_pairing_schedule is None:
				raise RuntimeError('RGB pairing schedule is unavailable during training.')
			rgb_pairing_indices = self._rgb_interventional_pairing_schedule.sample(
				root_id=rgb_interventional_batch.root_id.detach().cpu().numpy(),
				positive_code=(
					rgb_interventional_batch.positive_code.detach().cpu().numpy()
				),
				negative_code=(
					rgb_interventional_batch.negative_code.detach().cpu().numpy()
				),
			)
			kwargs.update(
				rgb_aux_clean_root=rgb_interventional_batch.clean_root,
				rgb_aux_hard_root=rgb_interventional_batch.hard_root,
				rgb_aux_clean_positive_future=(
					rgb_interventional_batch.clean_positive_future
				),
				rgb_aux_hard_positive_future=(
					rgb_interventional_batch.hard_positive_future
				),
				rgb_aux_clean_negative_future=(
					rgb_interventional_batch.clean_negative_future
				),
				rgb_aux_hard_negative_future=(
					rgb_interventional_batch.hard_negative_future
				),
				rgb_aux_action=rgb_interventional_batch.action,
				rgb_aux_negative_action=(
					rgb_interventional_batch.negative_action
				),
				rgb_aux_background_pairing=torch.as_tensor(
					rgb_pairing_indices.background,
					device=self.device, dtype=torch.int64,
				),
				rgb_aux_fork_pairing=torch.as_tensor(
					rgb_pairing_indices.fork,
					device=self.device, dtype=torch.int64,
				),
			)
			if self._rgb_interventional_aux_contract.get('arm') != 'data_matched':
				kwargs.update(
					rgb_aux_outcome_gap=rgb_interventional_batch.outcome_gap,
					rgb_aux_eligible=rgb_interventional_batch.eligible,
				)
		torch.compiler.cudagraph_mark_step_begin()
		info = self._update(obs, action, reward, terminated, **kwargs)
		if sibling_batch is not None:
			self._rof_real_sibling_aux_updates += 1
			self._rof_real_sibling_aux_samples += int(
				sibling_batch.action.shape[1]
			)
			info.update(
				sibling_aux_updates_total=torch.as_tensor(
					self._rof_real_sibling_aux_updates, device=self.device
				),
				sibling_aux_samples_total=torch.as_tensor(
					self._rof_real_sibling_aux_samples, device=self.device
				),
			)
		if rgb_interventional_batch is not None:
			self._rgb_interventional_aux_updates += 1
			self._rgb_interventional_aux_samples += int(
				rgb_interventional_batch.action.shape[1]
			)
			info.update(
				rgb_aux_updates_total=torch.as_tensor(
					self._rgb_interventional_aux_updates, device=self.device
				),
				rgb_aux_samples_total=torch.as_tensor(
					self._rgb_interventional_aux_samples, device=self.device
				),
				rgb_aux_pairing_calls_total=torch.as_tensor(
					self._rgb_interventional_pairing_schedule.metrics['calls'],
					device=self.device,
				),
				rgb_aux_background_correct_pairs_current=torch.as_tensor(
					rgb_pairing_indices.background_correct_pairs,
					device=self.device,
				),
				rgb_aux_fork_correct_pairs_current=torch.as_tensor(
					rgb_pairing_indices.fork_correct_pairs,
					device=self.device,
				),
			)
		if self._learned_object_belief:
			self._belief_update_calls += 1
			frequency = int(self.cfg.get('cutie_object_belief_update_frequency', 4))
			if frequency < 1:
				raise ValueError('cutie_object_belief_update_frequency must be positive.')
			if self._belief_update_calls % frequency == 0:
				belief_obs, belief_action = buffer.sample_belief()
				missing = self._sample_object_belief_missing(belief_obs)
				info.update(self._update_object_belief(
					belief_obs, belief_action, missing
				))
			else:
				zero = torch.zeros((), device=self.device)
				info.update(
					belief_loss=zero,
					belief_missing_loss=zero,
					belief_reacquisition_loss=zero,
					belief_reacquisition_teacher_roles=zero,
					belief_weighted_loss=zero,
					belief_grad_norm=zero,
					belief_teacher_role_steps=zero,
					belief_supervised_age_count=zero,
					belief_age20_loss=zero,
					belief_age20_teacher_roles=zero,
					belief_age20_supervision_available=zero,
					belief_reacquisition_supervision_available=zero,
					belief_pending_reacquisitions=zero,
					belief_synthetic_role_frames=zero,
					belief_affected_stack_roles=zero,
					belief_aux_attempt=zero,
					belief_aux_no_teacher_skip=zero,
					belief_aux_update=zero,
				)
		return info

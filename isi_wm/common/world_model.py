from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init
from common import cutie_object_belief
from common import cutie_object_auxiliary
from common import cutie_proprio
from common import gt_articulated_pose
from common import multimodal_articulated_pose
from common import mask_guided_rgb
from common import visual_articulated_pose
from common import object_state_supervision
from common import robust_object_field
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._cutie_regression_encoder = layers.validate_cutie_regression_encoder(cfg)
		requested_auxiliary_target = str(cfg.get(
			'cutie_object_auxiliary_target',
			cutie_object_auxiliary.FULL_DESCRIPTOR,
		))
		if requested_auxiliary_target not in cutie_object_auxiliary.TARGETS:
			raise ValueError(
				f'Unknown cutie_object_auxiliary_target '
				f'{requested_auxiliary_target!r}.'
			)
		if (
			requested_auxiliary_target
			== cutie_object_auxiliary.GEOMETRY_STATUS_FULL_DENOMINATOR
			and (
				not cfg.get('flat_anchor', False)
				or cfg.get('flat_anchor_mode', 'residual') not in {
					'cutie_hybrid', 'cutie_object_only',
				}
			)
		):
			raise ValueError(
				'geometry_status_full_denominator requires a Cutie object mode.'
			)
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout).apply(init.weight_init) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
		if mask_guided_rgb.enabled(cfg):
			self._install_mask_guided_rgb_branch()
		if cfg.get('flat_anchor', False):
			self._install_flat_anchor_branch()
		if object_state_supervision.enabled(cfg):
			object_state_supervision.validate_config(cfg)
			state_dim = object_state_supervision.target_dim(cfg)
			with torch.random.fork_rng(devices=[], enabled=True):
				self._state_supervision_head = layers.mlp(
					cfg.latent_dim, 128, state_dim,
				).apply(init.weight_init)
			if object_state_supervision.bottleneck_enabled(cfg):
				# Replace the object transition as well as the observation path.
				# Every control and imagination state is therefore a function of
				# predicted physical state and action only.
				with torch.random.fork_rng(devices=[], enabled=True):
					self._dynamics = layers.ObjectStateBottleneckDynamics(
						cfg, state_dim,
					).apply(init.weight_init)
				self._state_bottleneck_dim = state_dim

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def decode_supervised_state(self, z):
		"""Training-only state readout; privileged labels are never model inputs."""
		if object_state_supervision.bottleneck_enabled(self.cfg):
			return self._dynamics.unpack(z)
		return self._state_supervision_head(z)

	def encode_state_bottleneck(self, visual_latent):
		"""Discard visual latent after producing the deployable state estimate."""
		if not object_state_supervision.bottleneck_enabled(self.cfg):
			return visual_latent
		return self._dynamics.pack(self._state_supervision_head(visual_latent))

	def _install_flat_anchor_branch(self):
		"""Install the selected anchor path after official TD-MPC2 initialization."""
		mode = self.cfg.get('flat_anchor_mode', 'residual')
		if mode == 'residual':
			self._install_flat_anchor_residual_branch()
		elif mode == 'oc':
			self._install_oc_anchor_branch()
		elif mode == 'object_graph':
			self._install_object_graph_anchor_branch()
		elif mode == 'hybrid_graph':
			self._install_hybrid_graph_anchor_branch()
		elif mode == 'cutie_hybrid':
			self._install_cutie_hybrid_branch()
		elif mode == 'cutie_object_only':
			self._install_cutie_object_only_branch()
		elif mode == 'reward_graph':
			self._install_reward_graph_anchor_branch()
		else:
			raise ValueError(f'Unknown flat_anchor_mode: {mode!r}.')

	def _install_mask_guided_rgb_branch(self):
		"""Wrap the initialized official RGB encoder with the spatial residual."""
		mask_guided_rgb.validate_config(self.cfg)
		if self.cfg.get('flat_anchor', False):
			raise ValueError('Mask-guided RGB and FlatAnchor are mutually exclusive.')
		if set(self._encoder) != {'rgb'}:
			raise RuntimeError(
				'Mask-guided RGB must begin with exactly the official RGB encoder.'
			)
		with torch.random.fork_rng(devices=[], enabled=True):
			encoder = layers.MaskGuidedRGBEncoder(
				self._encoder['rgb'], self.cfg
			)
			# Initialize only the new mask branch. The captured official RGB modules
			# were already initialized and must remain byte-for-byte unchanged.
			encoder.role_stem.apply(init.weight_init)
			encoder.status_projection.apply(init.weight_init)
			encoder.film_output.apply(init.weight_init)
			encoder.reset_residual_output()
		self._encoder['rgb'] = encoder

	def _install_flat_anchor_residual_branch(self):
		"""Retain the earlier identity-residual path for completed ablations."""
		rgb_modules = list(self._encoder['rgb'].children())
		if not rgb_modules or not isinstance(rgb_modules[-1], layers.SimNorm):
			raise RuntimeError('Expected the official RGB encoder to end in SimNorm.')
		# SimNorm has no parameters. Removing it exposes the exact official RGB
		# logits; AnchorLogitFusion applies that same operation after modulation.
		self._encoder['rgb'] = nn.Sequential(*rgb_modules[:-1])

		# Module constructors and TD-MPC2's initializer both consume RNG. Forking
		# here leaves the official global RNG trajectory unchanged while giving the
		# new branch a deterministic, seed-specific initialization.
		with torch.random.fork_rng(devices=[], enabled=True):
			anchor = layers.mlp(
				self.cfg.flat_anchor_dim,
				self.cfg.flat_anchor_hidden_dim,
				self.cfg.flat_anchor_embed_dim,
			).apply(init.weight_init)
			fusion = layers.AnchorLogitFusion(self.cfg).apply(init.weight_init)
			init.zero_([fusion.projection.weight, fusion.projection.bias])
		self._encoder['anchor'] = anchor
		self._encoder['fusion'] = fusion

	def _install_oc_anchor_branch(self):
		"""Install separate scene/role encoding and OC-style token dynamics."""
		rgb_modules = list(self._encoder['rgb'].children())
		if not rgb_modules or not isinstance(rgb_modules[0], layers.ShiftAug):
			raise RuntimeError('Expected the official RGB encoder to start with ShiftAug.')
		# The original ShiftAug only transforms RGB. Replace it with a joint
		# RGB/anchor transform so role coordinates stay aligned with pixels.
		self._encoder['rgb'] = nn.Sequential(*rgb_modules[1:])
		self._oc_augmentation = layers.AnchorShiftAug(pad=rgb_modules[0].pad)

		# Construct all new trainable components in a forked CPU RNG stream so the
		# auxiliary branch itself does not advance the later global sampling stream.
		# OC heads use a wider 768-D state and are intentionally not parameter-wise
		# identical to the 512-D RGB baseline.
		with torch.random.fork_rng(devices=[], enabled=True):
			oc_encoder = layers.OCAnchorEncoder(self.cfg).apply(init.weight_init)
			oc_dynamics = layers.OCAnchorDynamics(self.cfg).apply(init.weight_init)
			anchor_decoder = layers.OCAnchorDecoder(self.cfg).apply(init.weight_init)
		self._encoder['oc'] = oc_encoder
		self._dynamics = oc_dynamics
		self._anchor_decoder = anchor_decoder

	def _install_object_graph_anchor_branch(self):
		"""Install the RGB-free four-role encoder and fixed-chain dynamics."""
		if len(self._encoder):
			raise RuntimeError('ObjectGraph must not construct an RGB/state encoder.')
		with torch.random.fork_rng(devices=[], enabled=True):
			anchor_encoder = layers.ObjectGraphAnchorEncoder(self.cfg).apply(init.weight_init)
			object_dynamics = layers.ObjectGraphDynamics(self.cfg).apply(init.weight_init)
			anchor_decoder = layers.ObjectGraphAnchorDecoder(self.cfg).apply(init.weight_init)
		self._encoder['anchor'] = anchor_encoder
		self._dynamics = object_dynamics
		self._anchor_decoder = anchor_decoder

	def _install_hybrid_graph_anchor_branch(self):
		"""Keep the exact RGB world model and add a lightweight role graph.

		Cross-modal information reaches the decision heads through zero-initialized
		correction networks. Therefore construction is exactly equivalent to the
		official RGB model while training can learn graph-conditioned reward, value,
		and policy predictions without running attention inside MPPI.
		"""
		rgb_modules = list(self._encoder['rgb'].children())
		if not rgb_modules or not isinstance(rgb_modules[0], layers.ShiftAug):
			raise RuntimeError('Expected the official RGB encoder to start with ShiftAug.')
		# Keep every subsequent module at its official state-dict index. Identity
		# has no state and the joint augmentation is applied immediately before it.
		augmentation_pad = rgb_modules[0].pad
		rgb_modules[0] = nn.Identity()
		self._encoder['rgb'] = nn.Sequential(*rgb_modules)
		self._oc_augmentation = layers.AnchorShiftAug(pad=augmentation_pad)

		scene_dim = self.cfg.flat_anchor_scene_dim
		graph_dim = self.cfg.flat_anchor_num_roles * self.cfg.get(
			'flat_anchor_object_role_dim', 32
		)
		joint_dim = self.cfg.get(
			'flat_anchor_hybrid_joint_dim', scene_dim + graph_dim
		)
		if self.cfg.latent_dim != scene_dim or joint_dim != scene_dim + graph_dim:
			raise ValueError(
				'HybridGraph must be installed from the official scene width and '
				'joint_dim must equal scene_dim+graph_dim.'
			)
		hidden_dim = self.cfg.get('flat_anchor_hybrid_hidden_dim', 128)
		output_dim = max(self.cfg.num_bins, 1)
		with torch.random.fork_rng(devices=[], enabled=True):
			anchor_encoder = layers.ObjectGraphAnchorEncoder(
				self.cfg, latent_dim=graph_dim
			).apply(init.weight_init)
			graph_dynamics = layers.ObjectGraphDynamics(
				self.cfg, latent_dim=graph_dim
			).apply(init.weight_init)
			graph_decoder = layers.ObjectGraphAnchorDecoder(
				self.cfg, latent_dim=graph_dim
			).apply(init.weight_init)
			anchor_decoder = layers.HybridGraphAnchorDecoder(
				graph_decoder, scene_dim, graph_dim
			)

			hybrid_reward = layers.HybridGraphCorrection(
				scene_dim, graph_dim, self.cfg.action_dim, hidden_dim, output_dim
			).apply(init.weight_init)
			hybrid_pi = layers.HybridGraphCorrection(
				scene_dim, graph_dim, 0, hidden_dim, 2 * self.cfg.action_dim
			).apply(init.weight_init)
			hybrid_termination = layers.HybridGraphCorrection(
				scene_dim, graph_dim, 0, hidden_dim, 1
			).apply(init.weight_init) if self.cfg.episodic else None
			hybrid_q_modules = []
			for _ in range(self.cfg.num_q):
				module = layers.HybridGraphCorrection(
					scene_dim, graph_dim, self.cfg.action_dim, hidden_dim, output_dim
				).apply(init.weight_init)
				init.zero_([module.output.weight, module.output.bias])
				hybrid_q_modules.append(module)
			hybrid_q = layers.Ensemble(hybrid_q_modules)

			# These branches are exact zeros at construction. The official modules,
			# their input shapes, and their floating-point kernels remain untouched.
			init.zero_([hybrid_reward.output.weight, hybrid_reward.output.bias])
			init.zero_([hybrid_pi.output.weight, hybrid_pi.output.bias])
			if hybrid_termination is not None:
				init.zero_([
					hybrid_termination.output.weight,
					hybrid_termination.output.bias,
				])

		self._encoder['anchor'] = anchor_encoder
		# Keep self._dynamics and every official state-dict key untouched.
		self._graph_dynamics = graph_dynamics
		self._anchor_decoder = anchor_decoder
		self._hybrid_reward = hybrid_reward
		self._hybrid_pi = hybrid_pi
		self._hybrid_termination = hybrid_termination
		self._hybrid_q = hybrid_q
		self._hybrid_scene_dim = scene_dim
		self._hybrid_graph_dim = graph_dim
		# TDMPC2 allocates planner/replay rollout tensors after model creation.
		self.cfg.latent_dim = joint_dim

	def _install_cutie_hybrid_branch(self):
		"""Add a generic two-object state while preserving official RGB step zero.

		Unlike the coordinate-anchor modes, the pooled Cutie observation is an
		independent state estimate and is therefore not transformed by pixel
		augmentation. The complete official RGB encoder (including ShiftAug), scene
		dynamics, and decision heads remain byte-for-byte and shape-for-shape intact.
		Zero-initialized corrections expose the object state to every decision head
		without changing their initial outputs.
		"""
		if 'rgb' not in self._encoder:
			raise RuntimeError('CutieHybrid requires the official RGB encoder.')
		rgb_modules = list(self._encoder['rgb'].children())
		if not rgb_modules or not isinstance(rgb_modules[0], layers.ShiftAug):
			raise RuntimeError('Expected the official RGB encoder to start with ShiftAug.')

		scene_dim = self.cfg.flat_anchor_scene_dim
		num_roles = self.cfg.get('cutie_object_num_roles', 2)
		role_dim = self.cfg.get('cutie_object_role_dim', 64)
		graph_dim = num_roles * role_dim
		joint_dim = self.cfg.get('cutie_object_joint_dim', scene_dim + graph_dim)
		if self.cfg.latent_dim != scene_dim:
			raise ValueError(
				'CutieHybrid must be installed from the official RGB latent width.'
			)
		if num_roles != 2 or graph_dim != 128 or joint_dim != scene_dim + graph_dim:
			raise ValueError(
				'CutieHybrid requires scene512 + two 64-D object roles = joint640.'
			)
		if tuple(self.cfg.obs_shape.get('object', ())) != (2, 1770):
			raise ValueError(
				"CutieHybrid requires observation key 'object' with shape (2, 1770)."
			)
		self._cutie_object_auxiliary_contract = cutie_object_auxiliary.contract(
			self.cfg.get('cutie_object_observation_variant', 'full'),
			self.cfg.get(
				'cutie_object_auxiliary_target',
				cutie_object_auxiliary.FULL_DESCRIPTOR,
			),
		)

		hidden_dim = self.cfg.get('flat_anchor_hybrid_hidden_dim', 128)
		output_dim = max(self.cfg.num_bins, 1)
		with torch.random.fork_rng(devices=[], enabled=True):
			object_encoder = layers.CutieObjectEncoder(self.cfg).apply(init.weight_init)
			object_dynamics = layers.CutieObjectDynamics(self.cfg).apply(init.weight_init)
			object_decoder = layers.CutieObjectDecoder(self.cfg).apply(init.weight_init)
			hybrid_reward = layers.HybridGraphCorrection(
				scene_dim, graph_dim, self.cfg.action_dim, hidden_dim, output_dim
			).apply(init.weight_init)
			hybrid_pi = layers.HybridGraphCorrection(
				scene_dim, graph_dim, 0, hidden_dim, 2 * self.cfg.action_dim
			).apply(init.weight_init)
			hybrid_termination = layers.HybridGraphCorrection(
				scene_dim, graph_dim, 0, hidden_dim, 1
			).apply(init.weight_init) if self.cfg.episodic else None
			hybrid_q_modules = []
			for _ in range(self.cfg.num_q):
				module = layers.HybridGraphCorrection(
					scene_dim, graph_dim, self.cfg.action_dim, hidden_dim, output_dim
				).apply(init.weight_init)
				init.zero_([module.output.weight, module.output.bias])
				hybrid_q_modules.append(module)
			hybrid_q = layers.Ensemble(hybrid_q_modules)

			init.zero_([hybrid_reward.output.weight, hybrid_reward.output.bias])
			init.zero_([hybrid_pi.output.weight, hybrid_pi.output.bias])
			if hybrid_termination is not None:
				init.zero_([
					hybrid_termination.output.weight,
					hybrid_termination.output.bias,
				])

		self._encoder['object'] = object_encoder
		self._graph_dynamics = object_dynamics
		self._object_decoder = object_decoder
		self._hybrid_reward = hybrid_reward
		self._hybrid_pi = hybrid_pi
		self._hybrid_termination = hybrid_termination
		self._hybrid_q = hybrid_q
		self._hybrid_scene_dim = scene_dim
		self._hybrid_graph_dim = graph_dim
		self.cfg.latent_dim = joint_dim

	def _install_cutie_object_only_branch(self):
		"""Install a structural Cutie controller state without a scene bypass.

		The legacy route receives descriptors only. ROF-WM additionally consumes
		causal RGB strictly through role-mask-gated spatial fields. In both cases,
		dynamics, reward, Q, policy, and termination operate on the role-complete
		state and no ordinary full-scene RGB encoder is constructed.
		"""
		if len(self._encoder):
			raise RuntimeError(
				'CutieObjectOnly must not construct an RGB or generic encoder.'
			)
		pose_oracle = gt_articulated_pose.enabled(self.cfg)
		visual_pose = visual_articulated_pose.enabled(self.cfg)
		multimodal_pose = multimodal_articulated_pose.enabled(self.cfg)
		cutie_proprio_variant = cutie_proprio.enabled(self.cfg)
		object_field_variant = robust_object_field.enabled(self.cfg)
		spatial_token_variant = bool(
			self.cfg.get('cutie_object_spatial_token_enabled', False)
		)
		variable_graph = bool(self.cfg.get(
			'cutie_object_variable_graph_enabled', False
		))
		pose_variant = pose_oracle or visual_pose or multimodal_pose
		if object_field_variant:
			robust_object_field.validate_config(self.cfg)
		if spatial_token_variant and (pose_variant or cutie_proprio_variant):
			raise ValueError(
				'Spatial-token control is a pure Full-Cutie observation mode.'
			)
		if spatial_token_variant and not self.cfg.get(
			'cutie_object_spatial_graph_path', None
		):
			raise ValueError('Spatial-token control requires an object graph path.')
		if variable_graph and not spatial_token_variant:
			raise ValueError('Variable object graphs require spatial-token control.')
		if variable_graph and (pose_variant or cutie_proprio_variant):
			raise ValueError('Variable object graphs are pure-visual full-descriptor models.')
		if pose_oracle:
			# The environment validates the same contract before simulator creation.
			# Repeating it here makes direct model construction fail closed too.
			gt_articulated_pose.validate_config(self.cfg)
		if visual_pose:
			visual_articulated_pose.validate_config(self.cfg)
		if multimodal_pose:
			multimodal_articulated_pose.validate_config(self.cfg)
		pose_contract = (
			gt_articulated_pose if pose_oracle else
			visual_articulated_pose if visual_pose else
			multimodal_articulated_pose
		)
		num_roles = int(self.cfg.get('cutie_object_num_roles', 2))
		expected_observation_shape = (
			(pose_contract.NUM_ROLES, pose_contract.INPUT_DIM)
			if pose_variant else (num_roles, 1770)
		)
		if cutie_proprio_variant:
			cutie_proprio.validate_config(self.cfg)
			expected_observation_shape = (
				cutie_proprio.NUM_ROLES, cutie_proprio.INPUT_DIM,
			)
		if object_field_variant:
			actual_shapes = {
				key: tuple(value) for key, value in dict(self.cfg.obs_shape).items()
			}
			if actual_shapes != robust_object_field.observation_shapes(self.cfg):
				raise ValueError(
					'ROF-WM observation shape contract mismatch: '
					f'{actual_shapes!r}.'
				)
		elif set(self.cfg.obs_shape) != {'object'} or tuple(
			self.cfg.obs_shape.get('object', ())
		) != expected_observation_shape:
			raise ValueError(
				"CutieObjectOnly requires only observation 'object' with shape "
				f'{expected_observation_shape}.'
			)
		role_dim = self.cfg.get('cutie_object_role_dim', 64)
		object_dim = robust_object_field.latent_dim(self.cfg) if object_field_variant else (
			(
				num_roles * role_dim
				if self.cfg.get('cutie_object_variable_graph_readout', 'pool') == 'direct'
				else int(self.cfg.get('cutie_object_only_latent_dim', 128))
			)
			if variable_graph else num_roles * role_dim
		)
		invalid_roles = False if object_field_variant else (
			not 1 <= num_roles <= int(self.cfg.get(
				'cutie_object_variable_graph_max_roles', 8
			)) if variable_graph else num_roles != layers.fixed_cutie_role_count(self.cfg)
		)
		if (
			invalid_roles or role_dim != 64
			or self.cfg.latent_dim != object_dim
			or self.cfg.get('cutie_object_only_latent_dim', 128) != object_dim
		):
			raise ValueError(
				'CutieObjectOnly requires 64-D role tokens and consistent model width.'
			)
		if cutie_proprio_variant:
			self._cutie_object_auxiliary_contract = cutie_proprio.auxiliary_contract(
				beta=self.cfg.get('flat_anchor_loss_beta', 0.1)
			)
		elif pose_variant:
			self._cutie_object_auxiliary_contract = (
				pose_contract.auxiliary_contract(
					beta=self.cfg.get('flat_anchor_loss_beta', 0.1)
				)
			)
		else:
			self._cutie_object_auxiliary_contract = cutie_object_auxiliary.contract(
				self.cfg.get('cutie_object_observation_variant', 'full'),
				self.cfg.get(
					'cutie_object_auxiliary_target',
					cutie_object_auxiliary.FULL_DESCRIPTOR,
				),
			)
		with torch.random.fork_rng(devices=[], enabled=True):
			if object_field_variant:
				object_encoder = robust_object_field.RobustObjectFieldEncoder(
					self.cfg
				).apply(init.weight_init)
				object_dynamics = robust_object_field.RobustObjectFieldDynamics(
					self.cfg
				).apply(init.weight_init)
				object_decoder = robust_object_field.RobustObjectFieldDecoder(
					self.cfg
				).apply(init.weight_init)
			else:
				object_encoder = layers.CutieObjectEncoder(self.cfg).apply(init.weight_init)
				object_encoder.reset_safe_fusion_output()
				object_dynamics = layers.CutieObjectDynamics(self.cfg).apply(init.weight_init)
				object_decoder = layers.CutieObjectDecoder(self.cfg).apply(init.weight_init)
			if self._cutie_regression_encoder == 'spatial_graph_v1':
				# Construct every shared module in the exact legacy RNG order first.
				# Replacing only the encoder after that keeps dynamics/decoder/head
				# initial parameters identical across all four regression arms.
				encoder_cfg = deepcopy(self.cfg)
				encoder_cfg.cutie_object_regression_encoder = None
				encoder_cfg.cutie_object_spatial_token_enabled = True
				encoder_cfg.cutie_object_variable_graph_enabled = True
				encoder_cfg.cutie_object_variable_graph_readout = 'direct'
				encoder_cfg.cutie_object_variable_graph_primary_skip_enabled = False
				object_encoder = layers.CutieObjectEncoder(encoder_cfg).apply(init.weight_init)
				object_encoder.reset_safe_fusion_output()
		belief_enabled = bool(self.cfg.get('cutie_object_belief_enabled', False))
		if variable_graph and belief_enabled:
			raise ValueError(
				'Legacy per-role belief is incompatible with pooled variable graphs.'
			)
		if belief_enabled:
			cutie_object_belief.validate_schema(self.cfg)
			if self.cfg.get('cutie_object_last_valid_memory', False):
				raise ValueError(
					'Learned object belief and wrapper last-valid memory are mutually exclusive.'
				)
			# Use the same initial parameterization as the world dynamics, then train
			# this transition independently with dedicated long-burst supervision.
			belief_dynamics = deepcopy(object_dynamics)
		self._encoder['object'] = object_encoder
		# Replace the generic latent MLP constructed before FlatAnchor dispatch.
		# The role-aware transition is the only imagined dynamics in this mode.
		self._dynamics = object_dynamics
		self._object_decoder = object_decoder
		self._object_latent_dim = object_dim
		if belief_enabled:
			self._belief_dynamics = belief_dynamics

	def belief_prior(self, belief, action):
		"""Predict the next causal object belief using the dedicated transition."""
		if not hasattr(self, '_belief_dynamics'):
			raise RuntimeError('This WorldModel has no learned object belief transition.')
		return self._belief_dynamics(belief, action)

	def belief_posterior(self, prior, measurement, objects):
		"""Correct visible roles exactly and retain priors only for missing roles."""
		valid = cutie_object_belief.latest_valid(objects)
		return cutie_object_belief.role_corrected_posterior(
			prior, measurement, valid
		), valid

	def _install_reward_graph_anchor_branch(self):
		"""Keep official control heads and add graph dynamics to reward only."""
		rgb_modules = list(self._encoder['rgb'].children())
		if not rgb_modules or not isinstance(rgb_modules[0], layers.ShiftAug):
			raise RuntimeError('Expected the official RGB encoder to start with ShiftAug.')
		# Preserve every parameterized RGB module and its state-dict index. The
		# replacement Identity lets one sampled crop be shared with role coordinates.
		augmentation_pad = rgb_modules[0].pad
		rgb_modules[0] = nn.Identity()
		self._encoder['rgb'] = nn.Sequential(*rgb_modules)
		self._oc_augmentation = layers.AnchorShiftAug(pad=augmentation_pad)

		scene_dim = self.cfg.flat_anchor_scene_dim
		graph_dim = self.cfg.flat_anchor_num_roles * self.cfg.get(
			'flat_anchor_object_role_dim', 32
		)
		joint_dim = self.cfg.get(
			'flat_anchor_hybrid_joint_dim', scene_dim + graph_dim
		)
		if self.cfg.latent_dim != scene_dim or joint_dim != scene_dim + graph_dim:
			raise ValueError(
				'RewardGraph must be installed from the official scene width and '
				'joint_dim must equal scene_dim+graph_dim.'
			)
		hidden_dim = self.cfg.get('flat_anchor_reward_graph_hidden_dim', 64)
		output_dim = max(self.cfg.num_bins, 1)
		with torch.random.fork_rng(devices=[], enabled=True):
			anchor_encoder = layers.ObjectGraphAnchorEncoder(
				self.cfg, latent_dim=graph_dim
			).apply(init.weight_init)
			graph_dynamics = layers.ObjectGraphDynamics(
				self.cfg, latent_dim=graph_dim
			).apply(init.weight_init)
			graph_decoder = layers.ObjectGraphAnchorDecoder(
				self.cfg, latent_dim=graph_dim
			).apply(init.weight_init)
			anchor_decoder = layers.HybridGraphAnchorDecoder(
				graph_decoder, scene_dim, graph_dim
			)
			# This correction has no scene input or bypass: the official scene
			# reward and graph reward are independent logits that are added later.
			graph_reward = layers.mlp(
				graph_dim + self.cfg.action_dim,
				hidden_dim,
				output_dim,
			).apply(init.weight_init)
			init.zero_([graph_reward[-1].weight, graph_reward[-1].bias])

		self._encoder['anchor'] = anchor_encoder
		self._graph_dynamics = graph_dynamics
		self._anchor_decoder = anchor_decoder
		self._graph_reward = graph_reward
		self._graph_scene_dim = scene_dim
		self._graph_latent_dim = graph_dim
		# Planner/replay state is scene512+graph128, while all official heads keep
		# their original 512-D construction and parameter shapes.
		self.cfg.latent_dim = joint_dim

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

		if self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'cutie_hybrid'}:
			self._detach_hybrid_q_params = TensorDictParams(
				self._hybrid_q.params.data, no_convert=True
			)
			self._target_hybrid_q_params = TensorDictParams(
				self._hybrid_q.params.data.clone(), no_convert=True
			)
			with self._detach_hybrid_q_params.data.to("meta").to_module(
				self._hybrid_q.module
			):
				self._detach_hybrid_q = deepcopy(self._hybrid_q)
				self._target_hybrid_q = deepcopy(self._hybrid_q)
			delattr(self._detach_hybrid_q, "params")
			self._detach_hybrid_q.__dict__["params"] = self._detach_hybrid_q_params
			delattr(self._target_hybrid_q, "params")
			self._target_hybrid_q.__dict__["params"] = self._target_hybrid_q_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		if self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'cutie_hybrid'}:
			self._target_hybrid_q.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)
		if self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'cutie_hybrid'}:
			self._target_hybrid_q_params.lerp_(
				self._detach_hybrid_q_params, self.cfg.tau
			)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def sample_oc_shift(self, batch_size, device):
		dtype = torch.float32 if self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'reward_graph'} else torch.int64
		return self._oc_augmentation.sample_shift(batch_size, device, dtype=dtype)

	def _encode_oc_frame(self, rgb, anchor, shift_index=None):
		rgb, anchor = self._oc_augmentation(rgb, anchor, shift_index=shift_index)
		rgb_latent = self._encoder['rgb'](rgb)
		return self._encoder['oc'](rgb_latent, anchor), anchor

	def _encode_hybrid_graph_frame(self, rgb, anchor, shift_index=None):
		rgb, anchor = self._oc_augmentation(rgb, anchor, shift_index=shift_index)
		scene = self._encoder['rgb'](rgb)
		graph = self._encoder['anchor'](anchor)
		return torch.cat([scene, graph], dim=-1), anchor

	def _encode_cutie_hybrid_frame(self, rgb, objects):
		# Preserve the complete official RGB augmentation and encoder. Pooled Cutie
		# descriptors are a separate observed state and are intentionally not shifted.
		scene = self._encoder['rgb'](rgb)
		graph = self._encoder['object'](objects)
		return torch.cat([scene, graph], dim=-1), objects

	def encode(
		self,
		obs,
		task,
		return_anchor=False,
		oc_shift=None,
		return_object=False,
	):
		"""
		Encodes an observation into its latent representation.
		Residual/OC modes consume RGB plus anchors; ObjectGraph consumes anchors only.
		"""
		if mask_guided_rgb.enabled(self.cfg):
			if return_anchor or return_object or oc_shift is not None:
				raise ValueError(
					'Mask-guided RGB does not expose anchor/object return paths.'
				)
			try:
				rgb = obs['rgb']
			except (KeyError, TypeError) as exc:
				raise ValueError(
					'Mask-guided RGB expects a structured observation.'
				) from exc
			if rgb.ndim == 5:
				for key in ('object_mask', 'tracker_status'):
					if obs[key].shape[:2] != rgb.shape[:2]:
						raise ValueError(
							f'Mask-guided sequence {key} leading dimensions do not match RGB.'
						)
				return torch.stack([
					self._encoder['rgb']({
						key: obs[key][index]
						for key in ('rgb', 'object_mask', 'tracker_status')
					})
					for index in range(rgb.shape[0])
				])
			if rgb.ndim == 4:
				return self._encoder['rgb'](obs)
			raise ValueError(
				'Mask-guided RGB must be [B,9,64,64] or [T,B,9,64,64].'
			)
		if self.cfg.get('flat_anchor', False):
			if self.cfg.multitask:
				raise NotImplementedError('FlatAnchor does not support multitask training.')
			mode = self.cfg.get('flat_anchor_mode', 'residual')
			if mode == 'cutie_object_only':
				if return_anchor:
					raise ValueError(
						'CutieObjectOnly uses return_object, not return_anchor.'
					)
				try:
					objects = obs['object']
				except (KeyError, TypeError) as exc:
					raise ValueError(
						"CutieObjectOnly expects only observation key 'object'."
					) from exc
				if robust_object_field.enabled(self.cfg):
					try:
						rgb = obs['rgb']
					except (KeyError, TypeError) as exc:
						raise ValueError(
							'ROF-WM observation is missing causal RGB.'
						) from exc
					if rgb.ndim == 5:
						for key in ('object', 'object_mask', 'role_exists'):
							if obs[key].shape[:2] != rgb.shape[:2]:
								raise ValueError(
									f'ROF sequence {key} leading dimensions do not match RGB.'
								)
						z = torch.stack([
							self._encoder['object']({
								key: obs[key][index]
								for key in ('rgb', 'object', 'object_mask', 'role_exists')
							})
							for index in range(rgb.shape[0])
						])
					elif rgb.ndim == 4:
						z = self._encoder['object'](obs)
					else:
						raise ValueError(
							'ROF RGB must be [B,9,64,64] or [T,B,9,64,64].'
						)
					return (z, objects) if return_object else z
				if objects.ndim not in {3, 4}:
					if cutie_proprio.enabled(self.cfg):
						expected_shape = '[B,2,1774] or [T,B,2,1774]'
					elif multimodal_articulated_pose.enabled(self.cfg):
						expected_shape = '[B,2,25] or [T,B,2,25]'
					elif (
						gt_articulated_pose.enabled(self.cfg)
						or visual_articulated_pose.enabled(self.cfg)
					):
						expected_shape = '[B,2,21] or [T,B,2,21]'
					else:
						expected_shape = '[B,2,1770] or [T,B,2,1770]'
					raise ValueError(
						f'CutieObjectOnly object tensor must have shape {expected_shape}, '
						f'got {tuple(objects.shape)}.'
					)
				z = self._encoder['object'](objects)
				z = self.encode_state_bottleneck(z)
				return (z, objects) if return_object else z
			if mode == 'cutie_hybrid':
				if return_anchor:
					raise ValueError('CutieHybrid uses return_object, not return_anchor.')
				try:
					rgb, objects = obs['rgb'], obs['object']
				except (KeyError, TypeError) as exc:
					raise ValueError(
						"CutieHybrid expects observation keys 'rgb' and 'object'."
					) from exc
				if rgb.ndim == 5:
					if objects.ndim != 4 or objects.shape[:2] != rgb.shape[:2]:
						raise ValueError(
							'CutieHybrid sequence RGB/object leading dimensions must match.'
						)
					encoded = [
						self._encode_cutie_hybrid_frame(frame, frame_objects)
						for frame, frame_objects in zip(
							rgb.unbind(0), objects.unbind(0)
						)
					]
					z = torch.stack([item[0] for item in encoded])
					used_objects = torch.stack([item[1] for item in encoded])
				elif rgb.ndim == 4:
					if objects.ndim != 3 or objects.shape[0] != rgb.shape[0]:
						raise ValueError(
							'CutieHybrid batch RGB/object leading dimensions must match.'
						)
					z, used_objects = self._encode_cutie_hybrid_frame(rgb, objects)
				else:
					raise ValueError(
						'CutieHybrid RGB must have shape [B,C,H,W] or [T,B,C,H,W], '
						f'got {tuple(rgb.shape)}.'
					)
				return (z, used_objects) if return_object else z
			if return_object:
				raise ValueError(
					'return_object is supported only by Cutie object modes.'
				)
			if mode == 'object_graph':
				try:
					anchor = obs['anchor']
				except (KeyError, TypeError) as exc:
					raise ValueError(
						"ObjectGraph expects an observation containing only 'anchor'."
					) from exc
				z = self._encoder['anchor'](anchor)
				return (z, anchor) if return_anchor else z

			try:
				rgb, anchor = obs['rgb'], obs['anchor']
			except (KeyError, TypeError) as exc:
				raise ValueError(
					"FlatAnchor expects an observation containing 'rgb' and 'anchor'."
				) from exc
			if mode in {'oc', 'hybrid_graph', 'reward_graph'}:
				encode_frame = (
					self._encode_oc_frame
					if mode == 'oc'
					else self._encode_hybrid_graph_frame
				)
				if rgb.ndim == 5:
					if mode in {'hybrid_graph', 'reward_graph'}:
						# Match official TD-MPC2 exactly: its RGB encoder invokes
						# ShiftAug once for every frame in the sequence. Each frame's
						# anchor must share that frame's shift, but shifts must not be
						# reused across time. An explicit [B,...] shift remains useful
						# for deterministic covariance tests; [T,B,...] supplies one
						# explicit shift per frame.
						if oc_shift is None:
							frame_shifts = [
								self.sample_oc_shift(rgb.shape[1], rgb.device)
								for _ in range(rgb.shape[0])
							]
						elif oc_shift.ndim == 5:
							if oc_shift.shape[:2] != rgb.shape[:2]:
								raise ValueError(
									'HybridGraph sequence shift must have leading shape '
									f'{tuple(rgb.shape[:2])}, got {tuple(oc_shift.shape[:2])}.'
								)
							frame_shifts = list(oc_shift.unbind(0))
						else:
							raise ValueError(
								'HybridGraph forbids reusing one [B,...] shift across a '
								'sequence; pass None or [T,B,1,1,2].'
							)
					else:
						if oc_shift is None:
							oc_shift = self.sample_oc_shift(rgb.shape[1], rgb.device)
						frame_shifts = [oc_shift] * rgb.shape[0]
					encoded = [
						encode_frame(frame, frame_anchor, frame_shift)
						for frame, frame_anchor, frame_shift in zip(
							rgb.unbind(0), anchor.unbind(0), frame_shifts
						)
					]
					z = torch.stack([item[0] for item in encoded])
					used_anchor = torch.stack([item[1] for item in encoded])
				elif rgb.ndim == 4:
					if oc_shift is None:
						oc_shift = self.sample_oc_shift(rgb.shape[0], rgb.device)
					z, used_anchor = encode_frame(rgb, anchor, oc_shift)
				else:
					raise ValueError(
						f'{mode} RGB must have shape [B,C,H,W] or [T,B,C,H,W], got {tuple(rgb.shape)}.'
					)
				return (z, used_anchor) if return_anchor else z

			if return_anchor:
				raise ValueError('return_anchor is only supported by structured anchor modes.')
			if rgb.ndim == 5:
				rgb_logits = torch.stack([self._encoder['rgb'](o) for o in rgb])
			elif rgb.ndim == 4:
				rgb_logits = self._encoder['rgb'](rgb)
			else:
				raise ValueError(
					f'FlatAnchor RGB must have shape [B,C,H,W] or [T,B,C,H,W], got {tuple(rgb.shape)}.'
				)
			if anchor.shape[:-1] != rgb_logits.shape[:-1]:
				raise ValueError(
					'FlatAnchor RGB and anchor leading dimensions must match, '
					f'got {tuple(rgb_logits.shape[:-1])} and {tuple(anchor.shape[:-1])}.'
				)
			anchor_z = self._encoder['anchor'](anchor)
			return self._encoder['fusion'](rgb_logits, anchor_z)

		if self.cfg.multitask:
			if return_object:
				raise ValueError('return_object is supported only by CutieHybrid mode.')
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def decode_anchor(self, z):
		"""Decode structured role sub-states into the teacher observation space."""
		if self.cfg.get('flat_anchor_mode', 'residual') not in {
			'oc', 'object_graph', 'hybrid_graph', 'reward_graph'
		}:
			raise RuntimeError('Anchor decoding requires OC or ObjectGraph mode.')
		return self._anchor_decoder(z)

	def anchor_loss(self, prediction, target):
		"""Confidence-weighted Smooth-L1 loss for structured anchors."""
		if prediction.shape != target.shape or prediction.shape[-1] != 25:
			raise ValueError('Anchor prediction and target must have matching [...,25] shapes.')
		confidence = target[..., 20:24].clamp(0., 1.)
		floor = self.cfg.flat_anchor_loss_weight_floor
		role_weight = floor + (1. - floor) * confidence
		position_weight = role_weight.unsqueeze(-1).expand(*role_weight.shape, 2).flatten(-2)
		edge_confidence = torch.stack([
			torch.minimum(confidence[..., 0], confidence[..., 1]),
			torch.minimum(confidence[..., 1], confidence[..., 2]),
			torch.minimum(confidence[..., 2], confidence[..., 3]),
		], dim=-1)
		edge_weight = (
			floor + (1. - floor) * edge_confidence
		).unsqueeze(-1).expand(*edge_confidence.shape, 2).flatten(-2)
		velocity_weight = role_weight[..., 1:].unsqueeze(-1).expand(
			*role_weight[..., 1:].shape, 2
		).flatten(-2)
		tail_weight = torch.ones_like(target[..., 20:25])
		weight = torch.cat([
			position_weight,
			edge_weight,
			velocity_weight,
			tail_weight,
		], dim=-1)
		element_loss = F.smooth_l1_loss(
			prediction,
			target,
			reduction='none',
			beta=self.cfg.flat_anchor_loss_beta,
		)
		return (element_loss * weight).sum() / weight.sum().clamp_min(1.)

	def decode_object(self, z):
		"""Decode Cutie roles from a hybrid joint state or object-only state."""
		mode = self.cfg.get('flat_anchor_mode', 'residual')
		if mode == 'cutie_hybrid':
			if z.shape[-1] != self._hybrid_scene_dim + self._hybrid_graph_dim:
				raise ValueError(
					'CutieHybrid decoder received an invalid joint latent width.'
				)
			return self._object_decoder(z[..., self._hybrid_scene_dim:])
		if mode == 'cutie_object_only':
			if z.shape[-1] != self._object_latent_dim:
				raise ValueError(
					'CutieObjectOnly decoder received an invalid latent width.'
				)
			return self._object_decoder(z)
		raise RuntimeError('Object decoding requires a Cutie object mode.')

	def object_loss(self, prediction, target):
		"""Validity-weighted auxiliary loss for two stacked Cutie objects.

		Each 1770-D role observation contains three 590-D frame descriptors. The
		latest descriptor's status is ``[confidence, lost, valid, mask_score]``;
		therefore its zero-based valid index is 588. Invalid latest roles retain a
		small floor weight so the model still learns the explicit missingness state.
		"""
		pose_oracle = gt_articulated_pose.enabled(self.cfg)
		visual_pose = visual_articulated_pose.enabled(self.cfg)
		multimodal_pose = multimodal_articulated_pose.enabled(self.cfg)
		if cutie_proprio.enabled(self.cfg):
			expected_tail = (cutie_proprio.NUM_ROLES, cutie_proprio.INPUT_DIM)
			if prediction.shape != target.shape or tuple(target.shape[-2:]) != expected_tail:
				raise ValueError(
					'Cutie-proprio prediction and target must have matching '
					f'[...,2,1774] shapes, got {tuple(prediction.shape)} and '
					f'{tuple(target.shape)}.'
				)
			beta = self.cfg.get('flat_anchor_loss_beta', 0.1)
			visual_prediction = prediction[..., :cutie_proprio.VISUAL_DIM]
			visual_target = target[..., :cutie_proprio.VISUAL_DIM]
			proprio_prediction = prediction[..., cutie_proprio.VISUAL_DIM:]
			proprio_target = target[..., cutie_proprio.VISUAL_DIM:]
			visual_element = F.smooth_l1_loss(
				visual_prediction, visual_target, reduction='none', beta=beta,
			).mean(dim=-1)
			latest_valid_index = 2 * 590 + 588
			valid = visual_target[..., latest_valid_index].clamp(0.0, 1.0)
			floor = float(self.cfg.get('flat_anchor_loss_weight_floor', 0.1))
			role_weight = floor + (1.0 - floor) * valid
			visual_loss = (
				visual_element * role_weight
			).sum() / role_weight.sum().clamp_min(1.0)
			proprio_loss = F.smooth_l1_loss(
				proprio_prediction, proprio_target, beta=beta,
			)
			selected_mode = cutie_proprio.mode(self.cfg)
			if selected_mode == 'cutie_only':
				return visual_loss
			if selected_mode == 'proprio_only':
				return proprio_loss
			return visual_loss + proprio_loss
		if pose_oracle or visual_pose or multimodal_pose:
			pose_contract = (
				gt_articulated_pose if pose_oracle else
				visual_articulated_pose if visual_pose else
				multimodal_articulated_pose
			)
			expected_tail = (
				pose_contract.NUM_ROLES,
				pose_contract.INPUT_DIM,
			)
			if (
				prediction.shape != target.shape
				or tuple(target.shape[-2:]) != expected_tail
			):
				raise ValueError(
					'Articulated-pose prediction and target must have matching '
					f'[...,{expected_tail[0]},{expected_tail[1]}] shapes, got '
					f'{tuple(prediction.shape)} and '
					f'{tuple(target.shape)}.'
				)
			expected_contract = pose_contract.auxiliary_contract(
				beta=self.cfg.get('flat_anchor_loss_beta', 0.1)
			)
			if self._cutie_object_auxiliary_contract != expected_contract:
				raise RuntimeError(
					'Articulated-pose auxiliary contract mismatch: '
					f'{self._cutie_object_auxiliary_contract!r} != '
					f'{expected_contract!r}.'
				)
			# Every pose value is finite and semantically supervised.  Unlike a
			# Cutie descriptor there is no query padding and no visibility/status
			# weighting, so ordinary elementwise SmoothL1 is the correct objective.
			return F.smooth_l1_loss(
				prediction,
				target,
				beta=self.cfg.get('flat_anchor_loss_beta', 0.1),
			)

		expected_tail = (
			int(self.cfg.get('cutie_object_num_roles', 2)),
			self.cfg.get('cutie_object_input_dim', 1770),
		)
		if prediction.shape != target.shape or tuple(target.shape[-2:]) != expected_tail:
			raise ValueError(
				'Cutie object prediction and target must have matching '
				f'[...,{expected_tail[0]},{expected_tail[1]}] shapes, got '
				f'{tuple(prediction.shape)} and '
				f'{tuple(target.shape)}.'
			)
		stack_frames = self.cfg.get('cutie_object_stack_frames', 3)
		frame_dim = self.cfg.get('cutie_object_frame_dim', 590)
		if stack_frames != 3 or frame_dim != 590 or expected_tail[-1] != stack_frames * frame_dim:
			raise ValueError('Cutie object loss requires three 590-D frame descriptors.')
		latest_valid_index = 2 * frame_dim + 588
		valid = target[..., latest_valid_index].clamp(0., 1.)
		floor = float(self.cfg.get('flat_anchor_loss_weight_floor', 0.1))
		if not 0. <= floor <= 1.:
			raise ValueError('flat_anchor_loss_weight_floor must lie in [0,1].')
		role_weight = floor + (1. - floor) * valid
		auxiliary = self._cutie_object_auxiliary_contract
		effective_target = auxiliary['effective_target']
		if effective_target == cutie_object_auxiliary.FULL_DESCRIPTOR:
			loss_prediction, loss_target = prediction, target
		elif effective_target == (
			cutie_object_auxiliary.LEGACY_GEOMETRY_STATUS_ACTIVE_DENOMINATOR
		):
			# The first 512 values of every 590-D frame are reserved exact zeros
			# in both geometry arms. Excluding them prevents 1536 trivial entries
			# per role from diluting supervision on the 74 geometry + 4 status
			# values that actually define this diagnostic observation schema.
			loss_prediction = prediction.reshape(
				*prediction.shape[:-1], stack_frames, frame_dim
			)[..., 512:]
			loss_target = target.reshape(
				*target.shape[:-1], stack_frames, frame_dim
			)[..., 512:]
		elif effective_target == (
			cutie_object_auxiliary.GEOMETRY_STATUS_FULL_DENOMINATOR
		):
			# Evaluate SmoothL1 elementwise on the unchanged 1770-D descriptor,
			# then mask only query supervision. Keeping the final mean over all
			# 1770 positions preserves the full-target per-element gradient scale.
			element_loss = F.smooth_l1_loss(
				prediction,
				target,
				reduction='none',
				beta=self.cfg.get('flat_anchor_loss_beta', 0.1),
			).reshape(*prediction.shape[:-1], stack_frames, frame_dim)
			feature_mask = element_loss.new_zeros(frame_dim)
			feature_mask[512:] = 1.
			element_loss = (
				element_loss * feature_mask
			).reshape(*prediction.shape[:-1], -1).mean(dim=-1)
			return (
				element_loss * role_weight
			).sum() / role_weight.sum().clamp_min(1.)
		else:  # Pure helper validation makes this unreachable.
			raise RuntimeError(f'Invalid Cutie auxiliary contract: {auxiliary!r}.')
		element_loss = F.smooth_l1_loss(
			loss_prediction,
			loss_target,
			reduction='none',
			beta=self.cfg.get('flat_anchor_loss_beta', 0.1),
		).reshape(*prediction.shape[:-1], -1).mean(dim=-1)
		return (element_loss * role_weight).sum() / role_weight.sum().clamp_min(1.)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'cutie_hybrid'}:
			scene = z[..., :self._hybrid_scene_dim]
			graph = z[..., self._hybrid_scene_dim:]
			next_scene = self._dynamics(torch.cat([scene, a], dim=-1))
			next_graph = self._graph_dynamics(graph, a)
			return torch.cat([next_scene, next_graph], dim=-1)
		if self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) == 'reward_graph':
			scene = z[..., :self._graph_scene_dim]
			graph = z[..., self._graph_scene_dim:]
			next_scene = self._dynamics(torch.cat([scene, a], dim=-1))
			next_graph = self._graph_dynamics(graph, a)
			return torch.cat([next_scene, next_graph], dim=-1)
		if (
			self.cfg.get('flat_anchor', False)
			and self.cfg.get('flat_anchor_mode', 'residual') in {
				'oc', 'object_graph', 'cutie_object_only'
			}
		):
			return self._dynamics(z, a)
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'cutie_hybrid'}:
			scene = z[..., :self._hybrid_scene_dim]
			return self._reward(torch.cat([scene, a], dim=-1)) + self._hybrid_reward(
				torch.cat([z, a], dim=-1)
			)
		if self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) == 'reward_graph':
			scene = z[..., :self._graph_scene_dim]
			graph = z[..., self._graph_scene_dim:]
			return self._reward(torch.cat([scene, a], dim=-1)) + self._graph_reward(
				torch.cat([graph, a], dim=-1)
			)
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		hybrid = self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'cutie_hybrid'}
		if hybrid:
			scene = z[..., :self._hybrid_scene_dim]
			logits = self._termination(scene) + self._hybrid_termination(z)
		elif self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) == 'reward_graph':
			logits = self._termination(z[..., :self._graph_scene_dim])
		elif self.cfg.multitask:
			z = self.task_emb(z, task)
			logits = self._termination(z)
		else:
			logits = self._termination(z)
		if unnormalized:
			return logits
		return torch.sigmoid(logits)
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		if self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'cutie_hybrid'}:
			scene = z[..., :self._hybrid_scene_dim]
			pi_logits = self._pi(scene) + self._hybrid_pi(z)
		elif self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) == 'reward_graph':
			pi_logits = self._pi(z[..., :self._graph_scene_dim])
		else:
			pi_logits = self._pi(z)
		mean, log_std = pi_logits.chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			# Keep every leaf on CUDA so this compiled partition remains eligible
			# for CUDA Graph replay. TensorDict would otherwise turn Python ``1.``
			# into a CPU scalar.
			"action_prob": log_prob.new_ones(()),
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		hybrid = self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) in {'hybrid_graph', 'cutie_hybrid'}
		if hybrid:
			scene_input = torch.cat([z[..., :self._hybrid_scene_dim], a], dim=-1)
			hybrid_input = torch.cat([z, a], dim=-1)
		elif self.cfg.get('flat_anchor', False) and self.cfg.get(
			'flat_anchor_mode', 'residual'
		) == 'reward_graph':
			z = torch.cat([z[..., :self._graph_scene_dim], a], dim=-1)
		else:
			z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
			hybrid_qnet = self._target_hybrid_q if hybrid else None
		elif detach:
			qnet = self._detach_Qs
			hybrid_qnet = self._detach_hybrid_q if hybrid else None
		else:
			qnet = self._Qs
			hybrid_qnet = self._hybrid_q if hybrid else None
		out = (
			qnet(scene_input) + hybrid_qnet(hybrid_input)
			if hybrid else qnet(z)
		)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2

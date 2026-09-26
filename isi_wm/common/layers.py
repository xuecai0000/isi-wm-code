import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy

from common import mask_guided_rgb, robust_object_field


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class AnchorShiftAug(nn.Module):
	"""Apply TD-MPC2's random shift jointly to RGB and anchor positions.

	A stacked RGB observation is translated as one image, so role positions move
	by the sampled translation while relative edges and temporal velocities stay
	unchanged. Keeping both modalities covariant avoids training on an augmented
	image paired with coordinates from the unaugmented frame.
	"""

	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def sample_shift(self, batch_size, device, dtype=torch.int64):
		return torch.randint(
			0,
			2 * self.pad + 1,
			size=(batch_size, 1, 1, 2),
			device=device,
			dtype=dtype,
		)

	def forward(self, x, anchor, shift_index=None):
		x = x.float()
		n, _, h, w = x.size()
		if h != w:
			raise ValueError('AnchorShiftAug expects square RGB observations.')
		if anchor.shape != (n, 25):
			raise ValueError(
				f'Expected anchor shape ({n}, 25), got {tuple(anchor.shape)}.'
			)
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(
			-1.0 + eps,
			1.0 - eps,
			h + 2 * self.pad,
			device=x.device,
			dtype=x.dtype,
		)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		if shift_index is None:
			shift_index = self.sample_shift(n, x.device)
		elif shift_index.shape != (n, 1, 1, 2):
			raise ValueError(
				f'Expected shared shift shape ({n},1,1,2), got {tuple(shift_index.shape)}.'
			)
		grid = base_grid + shift_index.to(x.dtype) * (2.0 / (h + 2 * self.pad))
		x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

		# The output crop samples padded pixels [shift, shift+h). An original
		# point therefore moves by (pad-shift) output pixels. Anchor coordinates
		# use [-1, 1] normalization with denominator (size-1).
		delta_pixels = self.pad - shift_index[:, 0, 0]
		delta = delta_pixels.to(anchor.dtype) * (2.0 / max(h - 1, 1))
		shifted_anchor = anchor.clone()
		positions = anchor[:, :8].reshape(n, 4, 2) + delta.unsqueeze(1)
		shifted_anchor[:, :8] = positions.reshape(n, 8)
		return x, shifted_anchor


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class AnchorLogitFusion(nn.Module):
	"""Identity-preserving FlatAnchor conditioning for a SimNorm latent.

	The projection is zeroed after the model-wide initializer runs. The RGB
	encoder provides the same pre-SimNorm logits as official TD-MPC2, so at
	initialization this computes the exact official SimNorm RGB path. Training
	can then use the anchor embedding as an additive logit modulation without
	replacing the RGB representation with a new bottleneck.
	"""

	def __init__(self, cfg):
		super().__init__()
		if cfg.flat_anchor_fusion_dim != cfg.latent_dim:
			raise ValueError(
				'Identity-preserving fusion requires flat_anchor_fusion_dim '
				f'({cfg.flat_anchor_fusion_dim}) to equal latent_dim ({cfg.latent_dim}).'
			)
		self.projection = nn.Linear(
			cfg.flat_anchor_embed_dim,
			cfg.flat_anchor_fusion_dim,
		)
		self.norm = SimNorm(cfg)

	def forward(self, rgb_logits, anchor_embedding):
		logits = rgb_logits + self.projection(anchor_embedding)
		return self.norm(logits)


class HybridGraphCorrection(nn.Module):
	"""Small balanced scene/graph fusion used as an additive head correction.

	The two modalities receive equal-width stems before interaction, so the
	128-D role graph cannot be numerically dominated by the 512-D scene merely
	because it has fewer coordinates. ``output`` is zero-initialized by the
	caller, leaving the complete official head exactly unchanged at step zero.
	"""

	def __init__(self, scene_dim, graph_dim, action_dim, hidden_dim, out_dim):
		super().__init__()
		if hidden_dim % 2:
			raise ValueError('HybridGraph hidden_dim must be even.')
		self.scene_dim = scene_dim
		self.graph_dim = graph_dim
		self.action_dim = action_dim
		branch_dim = hidden_dim // 2
		self.scene = NormedLinear(scene_dim + action_dim, branch_dim)
		self.graph = NormedLinear(graph_dim + action_dim, branch_dim)
		self.fusion = NormedLinear(hidden_dim, hidden_dim)
		self.output = nn.Linear(hidden_dim, out_dim)

	def forward(self, x):
		expected_dim = self.scene_dim + self.graph_dim + self.action_dim
		if x.shape[-1] != expected_dim:
			raise ValueError(
				f'HybridGraph correction expected input {expected_dim}, got {x.shape[-1]}.'
			)
		scene = x[..., :self.scene_dim]
		graph = x[..., self.scene_dim:self.scene_dim + self.graph_dim]
		if self.action_dim:
			action = x[..., -self.action_dim:]
			scene = torch.cat([scene, action], dim=-1)
			graph = torch.cat([graph, action], dim=-1)
		features = torch.cat([self.scene(scene), self.graph(graph)], dim=-1)
		return self.output(self.fusion(features))


LEGACY_MLP_K3_V1 = 'legacy_mlp_k3_v1'


def fixed_cutie_role_count(cfg):
	"""Return the cardinality admitted by a non-graph Cutie MLP mode."""
	return 3 if cfg.get('cutie_object_regression_encoder', None) == LEGACY_MLP_K3_V1 else 2


def validate_cutie_regression_encoder(cfg):
	"""Validate the narrow fixed-role encoder/auxiliary-target regression study.

	The public spatial/variable switches stay disabled so the live observation,
	decoder, transition and controller remain on the legacy path. Only the model
	installer builds a spatial encoder with a private configuration copy. The
	explicit K=3 legacy mode derives a 192-D state from three unchanged 64-D role
	tokens; the original null/legacy_mlp_v1 modes remain exactly K=2 and 128-D.
	"""
	mode = cfg.get('cutie_object_regression_encoder', None)
	if mode is None:
		return None
	if mode not in {'legacy_mlp_v1', 'spatial_graph_v1', LEGACY_MLP_K3_V1}:
		raise ValueError(f'Unknown Cutie regression encoder {mode!r}.')
	expected_roles = 3 if mode == LEGACY_MLP_K3_V1 else 2
	expected_latent = expected_roles * 64
	if (
		not cfg.get('flat_anchor', False)
		or cfg.get('flat_anchor_mode') != 'cutie_object_only'
		or cfg.get('multitask', False)
		or cfg.get('cutie_object_observation_variant', 'full') != 'full'
		or int(cfg.get('cutie_object_num_roles', 2)) != expected_roles
		or int(cfg.get('cutie_object_input_dim', 1770)) != 1770
		or int(cfg.get('cutie_object_frame_dim', 590)) != 590
		or int(cfg.get('cutie_object_stack_frames', 3)) != 3
		or int(cfg.get('cutie_object_role_dim', 64)) != 64
		or int(cfg.get('cutie_object_hidden_dim', 256)) != 256
		or int(cfg.get('cutie_object_only_latent_dim', 128)) != expected_latent
		or int(cfg.get('latent_dim', 128)) != expected_latent
	):
		raise ValueError(
			'Cutie regression requires legacy full '
			f'{expected_roles}-role {expected_latent}-D control for {mode}.'
		)
	for key in (
		'cutie_object_spatial_token_enabled',
		'cutie_object_variable_graph_enabled',
		'cutie_object_belief_enabled',
		'cutie_object_belief_use_for_control',
		'cutie_object_last_valid_memory',
		'cutie_object_true_entity_enabled',
		'object_state_supervision_enabled',
	):
		if cfg.get(key, False):
			raise ValueError(f'Cutie regression requires {key}=false.')
	if mode == 'spatial_graph_v1' and not cfg.get('cutie_object_spatial_graph_path'):
		raise ValueError('Spatial regression encoder requires its declared graph path.')
	if mode == 'spatial_graph_v1':
		graph = json.loads(Path(cfg.cutie_object_spatial_graph_path).read_text(encoding='utf-8'))
		if (
			graph.get('task') != cfg.get('task')
			or tuple(graph.get('source_roles', ()))
			!= tuple(cfg.get('cutie_object_role_names', ()))
		):
			raise ValueError('Regression graph must match the task and ordered input roles.')
	return mode


class CutieObjectEncoder(nn.Module):
	"""Encode ordered, history-stacked Cutie objects into compact role states.

	The full observation contract is ``[..., K, 1770]``. Each role contains three
	causal 590-D frame descriptors and is encoded by the same network. A one-hot
	role identity keeps configured roles semantically distinct without hard-coding
	task geometry into the transition model.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.num_roles = cfg.get('cutie_object_num_roles', 2)
		self.obs_dim = cfg.get('cutie_object_input_dim', 1770)
		self.observation_variant = cfg.get(
			'cutie_object_observation_variant', 'full'
		)
		self.role_dim = cfg.get('cutie_object_role_dim', 64)
		self.hidden_dim = cfg.get('cutie_object_hidden_dim', 256)
		self._spatial_token = bool(cfg.get('cutie_object_spatial_token_enabled', False))
		self._variable_graph = bool(cfg.get(
			'cutie_object_variable_graph_enabled', False
		))
		self.max_roles = int(cfg.get('cutie_object_variable_graph_max_roles', 8))
		self.pool_tokens = int(cfg.get('cutie_object_variable_graph_pool_tokens', 2))
		self.variable_graph_readout = str(cfg.get(
			'cutie_object_variable_graph_readout', 'pool'
		))
		self._primary_skip = bool(cfg.get(
			'cutie_object_variable_graph_primary_skip_enabled', False
		))
		if (
			not self._variable_graph
			and self.num_roles != fixed_cutie_role_count(cfg)
		):
			raise ValueError(
				'Cutie object encoder role count is incompatible with its fixed MLP mode.'
			)
		if self._variable_graph and not 1 <= self.num_roles <= self.max_roles:
			raise ValueError(
				'Variable object graph requires 1 <= num_roles <= max_roles, got '
				f'{self.num_roles} and {self.max_roles}.'
			)
		if self._variable_graph and not self._spatial_token:
			raise ValueError('Variable object graph requires spatial-token encoding.')
		if self._variable_graph and self.variable_graph_readout not in {'pool', 'direct'}:
			raise ValueError('Variable graph readout must be pool or direct.')
		pose_variants = {
			'gt_articulated_pose', 'visual_articulated_pose',
			'multimodal_articulated_pose',
		}
		expected_obs_dim = (
			1774 if self.observation_variant == 'cutie_proprio'
			else
			25 if self.observation_variant == 'multimodal_articulated_pose'
			else 21 if self.observation_variant in pose_variants else 1770
		)
		if self.observation_variant in pose_variants and cfg.get(
			'flat_anchor_mode', 'residual'
		) != 'cutie_object_only':
			raise ValueError('Articulated-pose variants are restricted to CutieObjectOnly.')
		if self.obs_dim != expected_obs_dim:
			raise ValueError(
				f'{self.observation_variant} requires {expected_obs_dim}-D stacked roles.'
			)
		self._cutie_proprio = self.observation_variant == 'cutie_proprio'
		self._cutie_proprio_mode = str(cfg.get('cutie_proprio_mode', 'fusion'))
		if self._spatial_token and (
			self.observation_variant != 'full' or self.obs_dim != 1770
		):
			raise ValueError('Spatial tokens require the full Kx1770 Cutie observation.')
		if self._spatial_token:
			token_dim = int(cfg.get('cutie_object_spatial_token_dim', 64))
			heads = int(cfg.get('cutie_object_spatial_num_heads', 4))
			layers_count = int(cfg.get('cutie_object_spatial_num_layers', 2))
			if (
				token_dim < 16 or token_dim % heads != 0
				or self.role_dim % heads != 0
			):
				raise ValueError(
					'Spatial token_dim and role_dim must be divisible by heads.'
				)
			if layers_count < 1:
				raise ValueError('Spatial token encoder requires at least one layer.')
			self.spatial_token_dim = token_dim
			self.patch_projection = nn.Linear(5, token_dim)
			self.query_projection = nn.Sequential(
				nn.Linear(512, token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
			)
			graph_path = Path(str(cfg.get('cutie_object_spatial_graph_path'))).resolve()
			graph = json.loads(graph_path.read_text(encoding='utf-8'))
			relations = graph.get('relations')
			if not isinstance(relations, list):
				raise ValueError('Spatial-token model requires a relation list.')
			relation_types = ('revolute_joint_v1', 'spatial_target_v1')
			if self._variable_graph:
				roles = tuple(graph.get('source_roles', ()))
				if len(roles) != self.num_roles or len(set(roles)) != self.num_roles:
					raise ValueError(
						'Variable graph source_roles must be unique and match num_roles.'
					)
				role_index = {name: index for index, name in enumerate(roles)}
				adjacency = torch.eye(self.num_roles, dtype=torch.bool)
				relation_context = torch.zeros(
					self.num_roles, 2 * len(relation_types), dtype=torch.float32,
				)
				for relation in relations:
					if not isinstance(relation, dict):
						raise ValueError('Every spatial relation must be an object.')
					try:
						parent = role_index[relation.get('parent')]
						child = role_index[relation.get('child')]
						type_index = relation_types.index(relation.get('type'))
					except (KeyError, ValueError) as exc:
						raise ValueError(f'Invalid spatial relation {relation!r}.') from exc
					if parent == child:
						raise ValueError('Spatial relations cannot be self edges.')
					adjacency[parent, child] = True
					adjacency[child, parent] = True
					relation_context[parent, type_index] += 1.0
					relation_context[child, len(relation_types) + type_index] += 1.0
				self.register_buffer(
					'_spatial_adjacency', adjacency, persistent=False,
				)
				self.register_buffer(
					'_spatial_relation_context', relation_context, persistent=False,
				)
				self.role_embedding = nn.Embedding(self.max_roles, token_dim)
				self.context_projection = nn.Sequential(
					nn.Linear(3 * 14 + token_dim + 2 * len(relation_types), token_dim),
					nn.LayerNorm(token_dim), nn.SiLU(),
				)
			else:
				if len(relations) != 1:
					raise ValueError('V1 spatial-token model requires exactly one edge.')
				try:
					relation_index = relation_types.index(relations[0].get('type'))
				except ValueError as exc:
					raise ValueError(
						f'Unsupported spatial relation {relations[0].get("type")!r}.'
					) from exc
				self.register_buffer(
					'_spatial_relation',
					F.one_hot(torch.tensor(relation_index), len(relation_types)).float(),
					persistent=True,
				)
				self.context_projection = nn.Sequential(
					nn.Linear(3 * 14 + self.num_roles + len(relation_types), token_dim),
					nn.LayerNorm(token_dim), nn.SiLU(),
				)
			block = nn.TransformerEncoderLayer(
				d_model=token_dim, nhead=heads,
				dim_feedforward=4 * token_dim, dropout=0.0,
				activation='gelu', batch_first=True, norm_first=True,
			)
			self.spatial_transformer = nn.TransformerEncoder(
				block, num_layers=layers_count, enable_nested_tensor=False,
			)
			self.spatial_output = nn.Sequential(
				nn.LayerNorm(token_dim), nn.Linear(token_dim, self.role_dim),
				SimNorm(cfg),
			)
			if self._variable_graph:
				expected_latent = (
					self.num_roles * self.role_dim
					if self.variable_graph_readout == 'direct'
					else self.pool_tokens * self.role_dim
				)
				if expected_latent != int(cfg.get('cutie_object_only_latent_dim', 128)):
					raise ValueError(
						'Variable graph readout width must equal object latent width.'
					)
				graph_block = nn.TransformerEncoderLayer(
					d_model=self.role_dim, nhead=heads,
					dim_feedforward=4 * self.role_dim, dropout=0.0,
					activation='gelu', batch_first=True, norm_first=True,
				)
				self.object_graph_transformer = nn.TransformerEncoder(
					graph_block, num_layers=layers_count, enable_nested_tensor=False,
				)
				if self.variable_graph_readout == 'pool':
					self.pool_queries = nn.Parameter(torch.empty(
						self.pool_tokens, self.role_dim
					))
					nn.init.normal_(self.pool_queries, std=0.02)
					self.object_pool = nn.MultiheadAttention(
						self.role_dim, heads, dropout=0.0, batch_first=True,
					)
					self.pool_output = nn.Sequential(
						nn.LayerNorm(self.pool_tokens * self.role_dim), SimNorm(cfg),
					)
					if self._primary_skip:
						self.pool_residual_gate_logit = nn.Parameter(torch.full(
							(self.pool_tokens, 1), -6.0
						))
				elif self._primary_skip:
					raise ValueError('Primary skip is only defined for pooled readout.')
			axis = torch.linspace(-1.0, 1.0, 8)
			yy, xx = torch.meshgrid(axis, axis, indexing='ij')
			self.register_buffer(
				'_spatial_xy', torch.stack([xx, yy], dim=-1).reshape(64, 2),
				persistent=False,
			)
		elif self._cutie_proprio:
			if self._cutie_proprio_mode not in {
				'cutie_only', 'proprio_only', 'fusion', 'factorized',
				'factorized_proprio_only',
			}:
				raise ValueError(f'Invalid cutie_proprio_mode {self._cutie_proprio_mode!r}.')
			self.role = mlp(
				1770 + self.num_roles, self.hidden_dim, self.role_dim,
				act=SimNorm(cfg),
			)
			self.proprio_role = mlp(
				4 + self.num_roles, self.hidden_dim, self.role_dim,
				act=SimNorm(cfg),
			)
			self.visual_delta = mlp(
				2 * self.role_dim, self.hidden_dim, self.role_dim,
			)
			self.fusion_gate_logit = nn.Parameter(
				torch.full((self.role_dim,), -4.0)
			)
			self.body_factor = mlp(
				self.num_roles * self.role_dim,
				self.hidden_dim,
				self.latent_dim // 2,
				act=SimNorm(cfg),
			)
			self.object_factor = mlp(
				self.num_roles * self.role_dim,
				self.hidden_dim,
				self.latent_dim // 2,
				act=SimNorm(cfg),
			)
		else:
			self.role = mlp(
				self.obs_dim + self.num_roles,
				self.hidden_dim,
				self.role_dim,
				act=SimNorm(cfg),
			)
		if not self._variable_graph:
			self.register_buffer('_role_ids', torch.eye(self.num_roles), persistent=False)

	def reset_safe_fusion_output(self):
		"""Restore the exact proprio-only initialization after global init."""
		if not self._cutie_proprio:
			return
		output = self.visual_delta[-1]
		if not isinstance(output, nn.Linear):
			raise RuntimeError('Safe visual-delta branch must end in Linear.')
		nn.init.zeros_(output.weight)
		nn.init.zeros_(output.bias)
		with torch.no_grad():
			self.fusion_gate_logit.fill_(-4.0)

	@property
	def latent_dim(self):
		if self._variable_graph:
			return (
				self.num_roles * self.role_dim
				if self.variable_graph_readout == 'direct'
				else self.pool_tokens * self.role_dim
			)
		return self.num_roles * self.role_dim

	def forward(self, objects):
		if tuple(objects.shape[-2:]) != (self.num_roles, self.obs_dim):
			raise ValueError(
				'Cutie object observation must have trailing shape '
				f'({self.num_roles}, {self.obs_dim}), got {tuple(objects.shape)}.'
			)
		lead = objects.shape[:-2]
		if not self._variable_graph:
			role_ids = self._role_ids.to(device=objects.device, dtype=objects.dtype)
			role_ids = role_ids.view(
				*([1] * len(lead)), self.num_roles, self.num_roles
			).expand(*lead, self.num_roles, self.num_roles)
		if self._spatial_token:
			frames = objects.reshape(*lead, self.num_roles, 3, 590)
			# Preserve each 8x8 cell as a token. Its local input is the causal
			# three-frame occupancy trace plus fixed image-plane coordinates.
			occupancy = frames[..., 512:576].transpose(-2, -1)
			xy = self._spatial_xy.to(device=objects.device, dtype=objects.dtype)
			xy = xy.view(
				*([1] * (len(lead) + 1)), 64, 2
			).expand(*lead, self.num_roles, 64, 2)
			patches = self.patch_projection(torch.cat([occupancy, xy], dim=-1))
			query = self.query_projection(frames[..., :512].mean(dim=-2))
			if self._variable_graph:
				indices = torch.arange(self.num_roles, device=objects.device)
				role_context = self.role_embedding(indices).to(dtype=objects.dtype)
				role_context = role_context.view(
					*([1] * len(lead)), self.num_roles, -1
				).expand(*lead, self.num_roles, -1)
				relation_context = self._spatial_relation_context.to(
					device=objects.device, dtype=objects.dtype,
				).view(*([1] * len(lead)), self.num_roles, -1).expand(
					*lead, self.num_roles, -1
				)
				context_values = torch.cat([
					frames[..., 576:590].flatten(start_dim=-2),
					role_context, relation_context,
				], dim=-1)
			else:
				relation = self._spatial_relation.to(
					device=objects.device, dtype=objects.dtype,
				).view(*([1] * (len(lead) + 1)), -1).expand(
					*lead, self.num_roles, -1
				)
				context_values = torch.cat([
					frames[..., 576:590].flatten(start_dim=-2), role_ids, relation,
				], dim=-1)
			context = query + self.context_projection(context_values)
			sequence = torch.cat([context.unsqueeze(-2), patches], dim=-2)
			encoded = self.spatial_transformer(
				sequence.reshape(-1, 65, self.spatial_token_dim)
			)[:, 0]
			roles = self.spatial_output(encoded).reshape(
				*lead, self.num_roles, self.role_dim
			)
			if self._variable_graph:
				flat_roles = roles.reshape(-1, self.num_roles, self.role_dim)
				attention_mask = ~self._spatial_adjacency.to(objects.device)
				graph_roles = self.object_graph_transformer(
					flat_roles, mask=attention_mask
				)
				if self.variable_graph_readout == 'direct':
					return graph_roles.reshape(
						*lead, self.num_roles * self.role_dim
					)
				queries = self.pool_queries.to(dtype=objects.dtype).unsqueeze(0).expand(
					graph_roles.shape[0], -1, -1
				)
				pooled, _ = self.object_pool(
					queries, graph_roles, graph_roles, need_weights=False
				)
				if self._primary_skip:
					# The graph's ordered primary roles retain an undiluted path. If
					# K is smaller than the fixed readout count, reuse the last real
					# role; this is not an empty or padded object token.
					indices = torch.arange(
						self.pool_tokens, device=objects.device
					).clamp_max(self.num_roles - 1)
					primary = flat_roles.index_select(1, indices)
					gate = torch.sigmoid(self.pool_residual_gate_logit).to(
						dtype=objects.dtype
					).unsqueeze(0)
					pooled = primary + gate * pooled
				pooled = pooled.reshape(*lead, self.pool_tokens * self.role_dim)
				return self.pool_output(pooled)
		elif self._cutie_proprio:
			visual = self.role(torch.cat([objects[..., :1770], role_ids], dim=-1))
			proprio = self.proprio_role(torch.cat([objects[..., 1770:], role_ids], dim=-1))
			if self._cutie_proprio_mode == 'cutie_only':
				roles = visual
			elif self._cutie_proprio_mode == 'proprio_only':
				roles = proprio
			elif self._cutie_proprio_mode in {
				'factorized', 'factorized_proprio_only',
			}:
				body = self.body_factor(proprio.flatten(start_dim=-2))
				scene = self.object_factor(visual.flatten(start_dim=-2))
				return torch.cat([body, scene], dim=-1)
			else:
				delta = self.visual_delta(torch.cat([proprio, visual], dim=-1))
				gate = torch.sigmoid(self.fusion_gate_logit).to(
					device=objects.device, dtype=objects.dtype,
				)
				roles = proprio + gate * delta
		else:
			roles = self.role(torch.cat([objects, role_ids], dim=-1))
		return roles.flatten(start_dim=-2)


class CutieObjectDynamics(nn.Module):
	"""Generic action-conditioned dynamics for the complete object-graph state."""

	def __init__(self, cfg):
		super().__init__()
		self.num_roles = cfg.get('cutie_object_num_roles', 2)
		self.role_dim = cfg.get('cutie_object_role_dim', 64)
		self.variable_graph = bool(cfg.get(
			'cutie_object_variable_graph_enabled', False
		))
		self.variable_graph_readout = str(cfg.get(
			'cutie_object_variable_graph_readout', 'pool'
		))
		self.latent_dim = (
			(
				self.num_roles * self.role_dim
				if self.variable_graph_readout == 'direct'
				else int(cfg.get('cutie_object_only_latent_dim', 128))
			)
			if self.variable_graph else self.num_roles * self.role_dim
		)
		self.action_dim = cfg.action_dim
		self.hidden_dim = cfg.get('cutie_object_hidden_dim', 256)
		if cfg.multitask:
			raise NotImplementedError('CutieHybrid currently supports single-task training only.')
		self.factorized = (
			cfg.get('cutie_object_observation_variant', 'full') == 'cutie_proprio'
			and cfg.get('cutie_proprio_mode', 'fusion') in {
				'factorized', 'factorized_proprio_only',
			}
		)
		if self.variable_graph and self.factorized:
			raise ValueError('Variable object graphs do not support proprio factorization.')
		if self.factorized:
			self.factor_dim = self.latent_dim // 2
			self.body_transition = mlp(
				self.factor_dim + self.action_dim,
				self.hidden_dim,
				self.factor_dim,
				act=SimNorm(cfg),
			)
			self.object_transition = mlp(
				2 * self.factor_dim + self.action_dim,
				self.hidden_dim,
				self.factor_dim,
				act=SimNorm(cfg),
			)
		else:
			self.transition = mlp(
				self.latent_dim + self.action_dim,
				self.hidden_dim,
				self.latent_dim,
				act=SimNorm(cfg),
			)

	def forward(self, z, action):
		if z.shape[-1] != self.latent_dim:
			raise ValueError(
				f'Cutie object latent must be {self.latent_dim}-D, got {z.shape[-1]}.'
			)
		if z.shape[:-1] != action.shape[:-1]:
			raise ValueError('Cutie object latent and action leading dimensions must match.')
		if self.factorized:
			body, scene = z.split(self.factor_dim, dim=-1)
			next_body = self.body_transition(torch.cat([body, action], dim=-1))
			next_scene = self.object_transition(
				torch.cat([scene, body, action], dim=-1)
			)
			return torch.cat([next_body, next_scene], dim=-1)
		return self.transition(torch.cat([z, action], dim=-1))


class ObjectStateBottleneckDynamics(nn.Module):
	"""Action-conditioned transition with no path around predicted state.

	Only the leading ``state_dim`` values of the 128-D TD-MPC2 latent may be
	non-zero. The remaining values are structural zero padding, so planning,
	reward, value and policy heads can consume the existing latent interface
	without receiving hidden visual features.
	"""

	def __init__(self, cfg, state_dim):
		super().__init__()
		self.latent_dim = int(cfg.latent_dim)
		self.state_dim = int(state_dim)
		self.action_dim = int(cfg.action_dim)
		self.hidden_dim = int(cfg.get('object_state_bottleneck_hidden_dim', 128))
		if cfg.multitask:
			raise NotImplementedError('Object-state bottleneck currently requires single-task training.')
		if not 0 < self.state_dim <= self.latent_dim:
			raise ValueError('Object-state bottleneck dimensions are invalid.')
		self.transition = mlp(
			self.state_dim + self.action_dim,
			self.hidden_dim,
			self.state_dim,
		)

	def pack(self, state):
		if state.shape[-1] != self.state_dim:
			raise ValueError(
				f'Predicted state must be {self.state_dim}-D, got {state.shape[-1]}.'
			)
		return F.pad(state, (0, self.latent_dim - self.state_dim))

	def unpack(self, latent):
		if latent.shape[-1] != self.latent_dim:
			raise ValueError(
				f'Control latent must be {self.latent_dim}-D, got {latent.shape[-1]}.'
			)
		return latent[..., :self.state_dim]

	def forward(self, latent, action):
		state = self.unpack(latent)
		if state.shape[:-1] != action.shape[:-1]:
			raise ValueError('Predicted state and action leading dimensions must match.')
		next_state = self.transition(torch.cat([state, action], dim=-1))
		return self.pack(next_state)


class CutieObjectDecoder(nn.Module):
	"""Decode compact role latents back to the frozen Cutie observation schema."""

	def __init__(self, cfg):
		super().__init__()
		self.num_roles = cfg.get('cutie_object_num_roles', 2)
		self.obs_dim = cfg.get('cutie_object_input_dim', 1770)
		self.observation_variant = cfg.get(
			'cutie_object_observation_variant', 'full'
		)
		self.role_dim = cfg.get('cutie_object_role_dim', 64)
		self.variable_graph = bool(cfg.get(
			'cutie_object_variable_graph_enabled', False
		))
		self.variable_graph_readout = str(cfg.get(
			'cutie_object_variable_graph_readout', 'pool'
		))
		self.max_roles = int(cfg.get('cutie_object_variable_graph_max_roles', 8))
		self.latent_dim = (
			(
				self.num_roles * self.role_dim
				if self.variable_graph_readout == 'direct'
				else int(cfg.get('cutie_object_only_latent_dim', 128))
			)
			if self.variable_graph else self.num_roles * self.role_dim
		)
		self.hidden_dim = cfg.get('cutie_object_hidden_dim', 256)
		pose_variants = {
			'gt_articulated_pose', 'visual_articulated_pose',
			'multimodal_articulated_pose',
		}
		expected_obs_dim = (
			1774 if self.observation_variant == 'cutie_proprio'
			else
			25 if self.observation_variant == 'multimodal_articulated_pose'
			else 21 if self.observation_variant in pose_variants else 1770
		)
		if self.observation_variant in pose_variants and cfg.get(
			'flat_anchor_mode', 'residual'
		) != 'cutie_object_only':
			raise ValueError('Articulated-pose variants are restricted to CutieObjectOnly.')
		if self.obs_dim != expected_obs_dim:
			raise ValueError(
				f'{self.observation_variant} requires {expected_obs_dim}-D decoder roles.'
			)
		self.factorized = (
			self.observation_variant == 'cutie_proprio'
			and cfg.get('cutie_proprio_mode', 'fusion') in {
				'factorized', 'factorized_proprio_only',
			}
		)
		if self.variable_graph:
			if self.observation_variant != 'full' or self.obs_dim != 1770:
				raise ValueError('Variable object graph decoder requires full descriptors.')
			if not 1 <= self.num_roles <= self.max_roles:
				raise ValueError('Variable object graph decoder role count is invalid.')
			self.role_embedding = nn.Embedding(self.max_roles, self.role_dim)
			self.role = mlp(
				self.latent_dim + self.role_dim,
				self.hidden_dim,
				self.obs_dim,
			)
		elif self.factorized:
			self.factor_dim = self.latent_dim // 2
			self.body_roles = mlp(
				self.factor_dim, self.hidden_dim,
				self.num_roles * self.role_dim,
			)
			self.object_roles = mlp(
				self.factor_dim, self.hidden_dim,
				self.num_roles * self.role_dim,
			)
			self.visual_role = mlp(
				self.role_dim + self.num_roles,
				self.hidden_dim, 1770,
			)
			self.proprio_role = mlp(
				self.role_dim + self.num_roles,
				self.hidden_dim, 4,
			)
		else:
			self.role = mlp(
				self.role_dim + self.num_roles,
				self.hidden_dim,
				self.obs_dim,
			)
		if not self.variable_graph:
			self.register_buffer('_role_ids', torch.eye(self.num_roles), persistent=False)

	def forward(self, z):
		if z.shape[-1] != self.latent_dim:
			raise ValueError(
				f'Cutie object decoder expects {self.latent_dim}-D, got {z.shape[-1]}.'
			)
		lead = z.shape[:-1]
		if self.variable_graph:
			indices = torch.arange(self.num_roles, device=z.device)
			role_context = self.role_embedding(indices).to(dtype=z.dtype)
			role_context = role_context.view(
				*([1] * len(lead)), self.num_roles, self.role_dim
			).expand(*lead, self.num_roles, self.role_dim)
			state = z.unsqueeze(-2).expand(*lead, self.num_roles, self.latent_dim)
			return self.role(torch.cat([state, role_context], dim=-1))
		roles = z.reshape(*lead, self.num_roles, self.role_dim)
		role_ids = self._role_ids.to(device=z.device, dtype=z.dtype)
		role_ids = role_ids.view(
			*([1] * len(lead)), self.num_roles, self.num_roles
		).expand(*lead, self.num_roles, self.num_roles)
		if self.factorized:
			body, scene = z.split(self.factor_dim, dim=-1)
			body_roles = self.body_roles(body).reshape(
				*lead, self.num_roles, self.role_dim
			)
			scene_roles = self.object_roles(scene).reshape(
				*lead, self.num_roles, self.role_dim
			)
			visual = self.visual_role(torch.cat([scene_roles, role_ids], dim=-1))
			proprio = self.proprio_role(torch.cat([body_roles, role_ids], dim=-1))
			return torch.cat([visual, proprio], dim=-1)
		return self.role(torch.cat([roles, role_ids], dim=-1))


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


class OCAnchorEncoder(nn.Module):
	"""Encode RGB context and role-structured anchors as separate sub-states.

	The returned tensor keeps the standard TD-MPC2 latent width, but its layout
	is explicit: ``[scene, base, elbow, control_tip, goal]``. This mirrors the
	object/visual separation used by OC-STORM without changing the public
	interfaces consumed by TD-MPC2's reward, Q, policy, and planner.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.num_roles = cfg.flat_anchor_num_roles
		self.scene_dim = cfg.flat_anchor_scene_dim
		self.role_dim = cfg.flat_anchor_role_dim
		self.role_input_dim = cfg.flat_anchor_role_input_dim
		if self.num_roles != 4:
			raise ValueError('The 25-D anchor contract contains exactly four roles.')
		if self.role_input_dim != 10:
			raise ValueError(
				'Each role input must contain position(2), velocity(2), '
				'incoming/outgoing edges(4), confidence(1), and fallback(1).'
			)
		if self.scene_dim + self.num_roles * self.role_dim != cfg.latent_dim:
			raise ValueError(
				'OC anchor sub-state dimensions must sum to latent_dim: '
				f'{self.scene_dim} + {self.num_roles}*{self.role_dim} '
				f'!= {cfg.latent_dim}.'
			)

		rgb_dim = 4 * 4 * cfg.num_channels
		self.scene = nn.Identity() if rgb_dim == self.scene_dim else mlp(
			rgb_dim,
			cfg.flat_anchor_hidden_dim,
			self.scene_dim,
			act=SimNorm(cfg),
		)
		# A shared object encoder follows OC-STORM. A one-hot role identity is
		# concatenated so persistent semantic roles cannot be permuted.
		self.role = mlp(
			self.role_input_dim + self.num_roles,
			cfg.flat_anchor_hidden_dim,
			self.role_dim,
			act=SimNorm(cfg),
		)
		self.register_buffer('_role_ids', torch.eye(self.num_roles), persistent=False)

	def role_inputs(self, anchor):
		"""Convert the flat 25-D observation into four role observations."""
		if anchor.shape[-1] != 25:
			raise ValueError(f'Expected a 25-D anchor observation, got {anchor.shape[-1]}.')
		lead = anchor.shape[:-1]
		positions = anchor[..., :8].reshape(*lead, self.num_roles, 2)
		edges = anchor[..., 8:14].reshape(*lead, 3, 2)
		moving_velocities = anchor[..., 14:20].reshape(*lead, 3, 2)
		velocities = torch.cat([
			torch.zeros_like(moving_velocities[..., :1, :]),
			moving_velocities,
		], dim=-2)
		confidence = anchor[..., 20:24].unsqueeze(-1)
		fallback = anchor[..., 24:25].unsqueeze(-2).expand(*lead, self.num_roles, 1)
		zero_edge = torch.zeros_like(edges[..., :1, :])
		incoming_edges = torch.cat([zero_edge, edges], dim=-2)
		outgoing_edges = torch.cat([edges, zero_edge], dim=-2)

		role_ids = self._role_ids.to(dtype=anchor.dtype)
		role_ids = role_ids.view(*([1] * len(lead)), self.num_roles, self.num_roles)
		role_ids = role_ids.expand(*lead, self.num_roles, self.num_roles)
		return torch.cat([
			positions,
			velocities,
			incoming_edges,
			outgoing_edges,
			confidence,
			fallback,
			role_ids,
		], dim=-1)

	def forward(self, rgb_latent, anchor):
		if rgb_latent.shape[:-1] != anchor.shape[:-1]:
			raise ValueError(
				'RGB and anchor leading dimensions must match, '
				f'got {tuple(rgb_latent.shape[:-1])} and {tuple(anchor.shape[:-1])}.'
			)
		scene = self.scene(rgb_latent)
		roles = self.role(self.role_inputs(anchor))
		roles = roles.flatten(start_dim=-2)
		return torch.cat([scene, roles], dim=-1)


class TokenSelfAttention(nn.Module):
	"""Small compile-friendly multi-head attention over object/scene tokens."""

	def __init__(self, dim, num_heads):
		super().__init__()
		if dim % num_heads:
			raise ValueError('Token dimension must be divisible by the number of heads.')
		self.dim = dim
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.qkv = nn.Linear(dim, 3 * dim, bias=False)
		self.proj = nn.Linear(dim, dim, bias=False)

	def forward(self, x):
		batch, tokens, _ = x.shape
		qkv = self.qkv(x).reshape(
			batch, tokens, 3, self.num_heads, self.head_dim
		).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)
		x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
		x = x.transpose(1, 2).reshape(batch, tokens, self.dim)
		return self.proj(x)


class OCTokenBlock(nn.Module):
	"""One OC-STORM-style spatial interaction block over K+1 tokens."""

	def __init__(self, dim, num_heads):
		super().__init__()
		self.attn_norm = nn.LayerNorm(dim)
		self.attn = TokenSelfAttention(dim, num_heads)
		self.ff_norm = nn.LayerNorm(dim)
		self.ff = mlp(dim, 2 * dim, dim)

	def forward(self, tokens):
		tokens = tokens + self.attn(self.attn_norm(tokens))
		return tokens + self.ff(self.ff_norm(tokens))


class OCAnchorDynamics(nn.Module):
	"""Action-conditioned object/scene interaction for one-step TD-MPC2 rollouts."""

	def __init__(self, cfg):
		super().__init__()
		if cfg.multitask:
			raise NotImplementedError('OC anchor dynamics currently supports single-task training only.')
		self.latent_dim = cfg.latent_dim
		self.action_dim = cfg.action_dim
		self.num_roles = cfg.flat_anchor_num_roles
		self.scene_dim = cfg.flat_anchor_scene_dim
		self.role_dim = cfg.flat_anchor_role_dim
		self.token_dim = cfg.flat_anchor_token_dim

		self.scene_stem = mlp(
			self.scene_dim + self.action_dim,
			self.token_dim,
			self.token_dim,
		)
		self.role_stem = mlp(
			self.role_dim + self.action_dim,
			self.token_dim,
			self.token_dim,
		)
		self.token_position = nn.Parameter(
			torch.zeros(1, self.num_roles + 1, self.token_dim)
		)
		self.blocks = nn.ModuleList([
			OCTokenBlock(self.token_dim, cfg.flat_anchor_num_heads)
			for _ in range(cfg.flat_anchor_num_layers)
		])
		self.scene_head = nn.Linear(self.token_dim, self.scene_dim)
		self.role_head = nn.Linear(self.token_dim, self.role_dim)
		self.scene_norm = SimNorm(cfg)
		self.role_norm = SimNorm(cfg)

	def split(self, z):
		scene = z[..., :self.scene_dim]
		roles = z[..., self.scene_dim:].reshape(
			*z.shape[:-1], self.num_roles, self.role_dim
		)
		return scene, roles

	def forward(self, z, action):
		if z.shape[:-1] != action.shape[:-1]:
			raise ValueError('Latent and action leading dimensions must match.')
		lead = z.shape[:-1]
		scene, roles = self.split(z)
		action_roles = action.unsqueeze(-2).expand(*lead, self.num_roles, self.action_dim)
		scene_token = self.scene_stem(torch.cat([scene, action], dim=-1)).unsqueeze(-2)
		role_tokens = self.role_stem(torch.cat([roles, action_roles], dim=-1))
		tokens = torch.cat([scene_token, role_tokens], dim=-2)

		# Flatten arbitrary leading dimensions for token attention. TD-MPC2 uses
		# both BxD and TxBxD tensors in its training and planning paths.
		tokens = tokens.reshape(-1, self.num_roles + 1, self.token_dim)
		tokens = tokens + self.token_position
		for block in self.blocks:
			tokens = block(tokens)

		next_scene = self.scene_norm(self.scene_head(tokens[:, 0]))
		next_roles = self.role_norm(self.role_head(tokens[:, 1:]))
		next_z = torch.cat([next_scene, next_roles.flatten(start_dim=-2)], dim=-1)
		return next_z.reshape(*lead, self.latent_dim)


class OCAnchorDecoder(nn.Module):
	"""Decode role sub-states back to the frozen teacher's 25-D observation."""

	def __init__(self, cfg):
		super().__init__()
		self.scene_dim = cfg.flat_anchor_scene_dim
		role_latent_dim = cfg.flat_anchor_num_roles * cfg.flat_anchor_role_dim
		self.decoder = mlp(
			role_latent_dim,
			cfg.flat_anchor_hidden_dim,
			cfg.flat_anchor_dim,
		)

	def forward(self, z):
		return self.decoder(z[..., self.scene_dim:])


class ObjectGraphAnchorEncoder(nn.Module):
	"""Encode the 25-D anchor observation into four compact role states.

	Unlike the OC path, this encoder deliberately has no scene/RGB state. Each
	role receives its position, velocity, incident chain edges, confidence,
	fallback flag, and a persistent one-hot role identity. A shared encoder keeps
	the object path small while the role identity prevents semantic permutation.
	"""

	def __init__(self, cfg, latent_dim=None):
		super().__init__()
		self.num_roles = cfg.flat_anchor_num_roles
		self.role_dim = cfg.get('flat_anchor_object_role_dim', 32)
		self.hidden_dim = cfg.get('flat_anchor_object_hidden_dim', 64)
		self.latent_dim = cfg.latent_dim if latent_dim is None else latent_dim
		if self.num_roles != 4:
			raise ValueError('ObjectGraph requires the four roles base/elbow/tip/goal.')
		if self.num_roles * self.role_dim != self.latent_dim:
			raise ValueError(
				'ObjectGraph latent_dim must equal num_roles*object_role_dim: '
				f'{self.latent_dim} != {self.num_roles}*{self.role_dim}.'
			)
		self.role = mlp(
			10 + self.num_roles,
			self.hidden_dim,
			self.role_dim,
			act=SimNorm(cfg),
		)
		self.register_buffer('_role_ids', torch.eye(self.num_roles), persistent=False)

	def role_inputs(self, anchor):
		if anchor.shape[-1] != 25:
			raise ValueError(f'Expected a 25-D anchor observation, got {anchor.shape[-1]}.')
		lead = anchor.shape[:-1]
		positions = anchor[..., :8].reshape(*lead, self.num_roles, 2)
		edges = anchor[..., 8:14].reshape(*lead, 3, 2)
		moving_velocities = anchor[..., 14:20].reshape(*lead, 3, 2)
		velocities = torch.cat([
			torch.zeros_like(moving_velocities[..., :1, :]),
			moving_velocities,
		], dim=-2)
		confidence = anchor[..., 20:24].unsqueeze(-1)
		fallback = anchor[..., 24:25].unsqueeze(-2).expand(*lead, self.num_roles, 1)
		zero_edge = torch.zeros_like(edges[..., :1, :])
		incoming_edges = torch.cat([zero_edge, edges], dim=-2)
		outgoing_edges = torch.cat([edges, zero_edge], dim=-2)

		role_ids = self._role_ids.to(dtype=anchor.dtype)
		role_ids = role_ids.view(*([1] * len(lead)), self.num_roles, self.num_roles)
		role_ids = role_ids.expand(*lead, self.num_roles, self.num_roles)
		return torch.cat([
			positions,
			velocities,
			incoming_edges,
			outgoing_edges,
			confidence,
			fallback,
			role_ids,
		], dim=-1)

	def forward(self, anchor):
		roles = self.role(self.role_inputs(anchor))
		return roles.flatten(start_dim=-2)


class ObjectGraphDynamics(nn.Module):
	"""Cheap action-conditioned message passing on base-elbow-tip-goal.

	Messages only traverse the fixed mechanical chain. The action is exposed to
	the controllable elbow and tip roles; base and goal receive exact zeros in
	the action channels. There is no attention, scene token, or dense all-to-all
	interaction inside MPPI rollouts.
	"""

	def __init__(self, cfg, latent_dim=None):
		super().__init__()
		if cfg.multitask:
			raise NotImplementedError('ObjectGraph dynamics supports single-task training only.')
		self.num_roles = cfg.flat_anchor_num_roles
		self.role_dim = cfg.get('flat_anchor_object_role_dim', 32)
		self.hidden_dim = cfg.get('flat_anchor_object_hidden_dim', 64)
		self.action_dim = cfg.action_dim
		self.latent_dim = cfg.latent_dim if latent_dim is None else latent_dim
		if self.num_roles != 4:
			raise ValueError('ObjectGraph requires four ordered roles.')
		if self.latent_dim != self.num_roles * self.role_dim:
			raise ValueError('ObjectGraph latent layout must be [4, object_role_dim].')

		# The same local transition is shared across roles. Role IDs distinguish
		# semantics; zero-padded neighbours expose chain endpoints explicitly.
		self.transition = mlp(
			3 * self.role_dim + self.action_dim + self.num_roles,
			self.hidden_dim,
			self.role_dim,
			act=SimNorm(cfg),
		)
		self.register_buffer('_role_ids', torch.eye(self.num_roles), persistent=False)
		self.register_buffer(
			'_action_role_mask',
			torch.tensor([0., 1., 1., 0.]).view(1, self.num_roles, 1),
			persistent=False,
		)

	def forward(self, z, action):
		if z.shape[:-1] != action.shape[:-1]:
			raise ValueError('Latent and action leading dimensions must match.')
		lead = z.shape[:-1]
		roles = z.reshape(*lead, self.num_roles, self.role_dim)
		zero = torch.zeros_like(roles[..., :1, :])
		previous_role = torch.cat([zero, roles[..., :-1, :]], dim=-2)
		next_role = torch.cat([roles[..., 1:, :], zero], dim=-2)

		action_per_role = action.unsqueeze(-2).expand(
			*lead, self.num_roles, self.action_dim
		)
		mask = self._action_role_mask.to(dtype=action.dtype)
		mask = mask.view(*([1] * len(lead)), self.num_roles, 1)
		action_per_role = action_per_role * mask

		role_ids = self._role_ids.to(dtype=z.dtype)
		role_ids = role_ids.view(*([1] * len(lead)), self.num_roles, self.num_roles)
		role_ids = role_ids.expand(*lead, self.num_roles, self.num_roles)
		transition_input = torch.cat([
			previous_role,
			roles,
			next_role,
			action_per_role,
			role_ids,
		], dim=-1)
		next_roles = self.transition(transition_input)
		return next_roles.flatten(start_dim=-2)


class ObjectGraphAnchorDecoder(nn.Module):
	"""Decode the compact four-role state to the 25-D teacher observation."""

	def __init__(self, cfg, latent_dim=None):
		super().__init__()
		self.latent_dim = cfg.latent_dim if latent_dim is None else latent_dim
		self.decoder = mlp(
			self.latent_dim,
			cfg.get('flat_anchor_object_hidden_dim', 64),
			cfg.flat_anchor_dim,
		)

	def forward(self, z):
		return self.decoder(z)


class HybridGraphAnchorDecoder(nn.Module):
	"""Decode only the persistent graph portion of a hybrid latent state."""

	def __init__(self, decoder, scene_dim, graph_dim):
		super().__init__()
		self.decoder = decoder
		self.scene_dim = scene_dim
		self.graph_dim = graph_dim

	def forward(self, z):
		if z.shape[-1] != self.scene_dim + self.graph_dim:
			raise ValueError('HybridGraph decoder received an invalid latent width.')
		return self.decoder(z[..., self.scene_dim:])


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


class JointMaskShiftAug(nn.Module):
	"""Apply the official TD-MPC2 random crop to RGB and Cutie masks jointly."""

	def __init__(self, pad=3):
		super().__init__()
		self.pad = int(pad)
		self.padding = (self.pad,) * 4

	def forward(self, rgb, masks, shift_index=None):
		if rgb.ndim != 4 or masks.ndim != 5:
			raise ValueError(
				'Mask-guided shift expects RGB [B,9,H,W] and masks [B,K,3,H,W].'
			)
		batch, _, height, width = rgb.shape
		if (
			height != width
			or masks.shape[0] != batch
			or tuple(masks.shape[-2:]) != (height, width)
		):
			raise ValueError('Mask-guided RGB and masks must share square image dimensions.')
		rgb = rgb.float()
		masks = masks.float()
		if self.pad == 0:
			return rgb, masks
		rgb_pad = F.pad(rgb, self.padding, mode='replicate')
		flat_masks = masks.reshape(batch, -1, height, width)
		mask_pad = F.pad(flat_masks, self.padding, mode='constant', value=0.0)
		padded = height + 2 * self.pad
		eps = 1.0 / padded
		axis = torch.linspace(
			-1.0 + eps, 1.0 - eps, padded,
			device=rgb.device, dtype=rgb.dtype,
		)[:height]
		axis = axis.unsqueeze(0).repeat(height, 1).unsqueeze(2)
		grid = torch.cat([axis, axis.transpose(1, 0)], dim=2)
		grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
		if shift_index is None:
			shift_index = torch.randint(
				0, 2 * self.pad + 1, (batch, 1, 1, 2),
				device=rgb.device, dtype=rgb.dtype,
			)
		elif tuple(shift_index.shape) != (batch, 1, 1, 2):
			raise ValueError(
				'Mask-guided shift_index must have shape '
				f'[{batch},1,1,2], got {tuple(shift_index.shape)}.'
			)
		grid = grid + shift_index.to(grid.dtype) * (2.0 / padded)
		shifted_rgb = F.grid_sample(
			rgb_pad, grid, padding_mode='zeros', align_corners=False
		)
		shifted_masks = F.grid_sample(
			mask_pad, grid, mode='nearest', padding_mode='zeros', align_corners=False
		).reshape_as(masks)
		return shifted_rgb, shifted_masks


class MaskGuidedRGBEncoder(nn.Module):
	"""Official RGB CNN plus a bounded, zero-initialized spatial Mask residual.

	The exact official convolution path and its parameters are retained. A shared
	per-role mask stem is pooled over the real role axis, then predicts spatial
	FiLM scale/bias corrections at the official 4x4 feature map. The output layer
	is reset to zero after initialization, making step zero exactly the RGB model.
	"""

	def __init__(self, official_encoder, cfg):
		super().__init__()
		mask_guided_rgb.validate_config(cfg)
		modules = list(official_encoder.children())
		if (
			len(modules) != 11
			or not isinstance(modules[0], ShiftAug)
			or not isinstance(modules[1], PixelPreprocess)
			or not isinstance(modules[9], nn.Flatten)
			or not isinstance(modules[10], SimNorm)
		):
			raise RuntimeError(
				'Mask-guided RGB requires the untouched official TD-MPC2 pixel encoder.'
			)
		self.num_roles = int(cfg.get('cutie_object_num_roles', 0))
		self.hidden_channels = int(cfg.get(
			'cutie_mask_guided_rgb_hidden_channels', 32
		))
		self.correction_limit = float(cfg.get(
			'cutie_mask_guided_rgb_correction_limit', 0.25
		))
		self.ablation_mode = str(cfg.get(
			'cutie_mask_guided_rgb_ablation_mode', 'none'
		))
		permutation_generator = torch.Generator(device='cpu')
		permutation_generator.manual_seed(
			mask_guided_rgb.SPATIAL_PERMUTATION_SEED
		)
		self.register_buffer(
			'_mask_spatial_permutation',
			torch.randperm(
				mask_guided_rgb.IMAGE_SIZE ** 2,
				generator=permutation_generator,
			),
		)
		self.augmentation = JointMaskShiftAug(
			cfg.get('cutie_mask_guided_rgb_random_shift_pad', 3)
		)
		# Reuse the already initialized official modules without reconstructing them.
		self.rgb_preprocess = modules[1]
		self.rgb_spatial = nn.Sequential(*modules[2:9])
		self.rgb_flatten = modules[9]
		self.rgb_norm = modules[10]
		mid = max(16, self.hidden_channels // 2)
		# Per role/frame: the raw mask plus one mask-weighted map for each
		# tracker status scalar. A separate status projection preserves explicit
		# missingness evidence even when a predicted mask is empty.
		self.role_stem = nn.Sequential(
			nn.Conv2d(
				(1 + mask_guided_rgb.STATUS_DIM) * mask_guided_rgb.STACK_FRAMES,
				mid, 7, stride=2,
			),
			nn.ReLU(inplace=False),
			nn.Conv2d(mid, self.hidden_channels, 5, stride=2),
			nn.ReLU(inplace=False),
			nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, stride=2),
			nn.ReLU(inplace=False),
			nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, stride=1),
			nn.ReLU(inplace=False),
		)
		self.status_projection = nn.Linear(
			mask_guided_rgb.STACK_FRAMES * mask_guided_rgb.STATUS_DIM,
			self.hidden_channels,
		)
		self.film_output = nn.Conv2d(
			self.hidden_channels, 2 * int(cfg.num_channels), 1
		)

	def reset_residual_output(self):
		nn.init.zeros_(self.film_output.weight)
		nn.init.zeros_(self.film_output.bias)

	def _ablate_masks(self, masks):
		if self.ablation_mode == 'spatial_permute_v1':
			# A deterministic bijection preserves every mask bit and exact area,
			# while destroying local shape and registration to the paired RGB.
			return masks.flatten(-2).index_select(
				-1, self._mask_spatial_permutation
			).reshape_as(masks)
		return masks

	def _validate(self, observation):
		try:
			rgb = observation['rgb']
			masks = observation['object_mask']
			status = observation['tracker_status']
		except (KeyError, TypeError) as exc:
			raise ValueError(
				'Mask-guided encoder requires rgb/object_mask/tracker_status.'
			) from exc
		batch = rgb.shape[0] if rgb.ndim == 4 else None
		expected = {
			'rgb': (batch, mask_guided_rgb.RGB_CHANNELS,
				mask_guided_rgb.IMAGE_SIZE, mask_guided_rgb.IMAGE_SIZE),
			'object_mask': (batch, self.num_roles, mask_guided_rgb.STACK_FRAMES,
				mask_guided_rgb.IMAGE_SIZE, mask_guided_rgb.IMAGE_SIZE),
			'tracker_status': (batch, self.num_roles, mask_guided_rgb.STACK_FRAMES,
				mask_guided_rgb.STATUS_DIM),
		}
		actual = {
			'rgb': tuple(rgb.shape), 'object_mask': tuple(masks.shape),
			'tracker_status': tuple(status.shape),
		}
		if rgb.ndim != 4 or actual != expected:
			raise ValueError(
				f'Mask-guided observation shape mismatch: {actual} != {expected}.'
			)
		if rgb.dtype != torch.uint8 or masks.dtype != torch.bool:
			raise ValueError('Mask-guided RGB must be uint8 and masks must be bool.')
		if not torch.is_floating_point(status):
			raise ValueError('Mask-guided tracker status must be floating point.')
		if not torch.compiler.is_compiling():
			if not bool(torch.isfinite(status).all().item()):
				raise ValueError('Mask-guided tracker status must be finite.')
			confidence = status[..., 0]
			lost = status[..., 1]
			valid = status[..., 2]
			mask_score = status[..., 3]
			if not bool(((confidence >= 0.0) & (confidence <= 1.0)).all().item()):
				raise ValueError('Mask-guided confidence must lie in [0,1].')
			if not bool(((lost == 0.0) | (lost == 1.0)).all().item()):
				raise ValueError('Mask-guided lost status must be binary.')
			if not bool(((valid == 0.0) | (valid == 1.0)).all().item()):
				raise ValueError('Mask-guided validity must be binary.')
			if not bool(((mask_score >= 0.0) & (mask_score <= 1.0)).all().item()):
				raise ValueError('Mask-guided mask score must lie in [0,1].')
		return rgb, masks, status

	def forward(self, observation, *, shift_index=None, return_correction=False):
		rgb, masks, status = self._validate(observation)
		rgb, masks = self.augmentation(rgb, masks, shift_index=shift_index)
		masks = self._ablate_masks(masks)
		spatial = self.rgb_spatial(self.rgb_preprocess(rgb))
		batch, roles = masks.shape[:2]
		status_maps = status.unsqueeze(-1).unsqueeze(-1) * masks.unsqueeze(3)
		role_input = torch.cat([
			masks.unsqueeze(3), status_maps,
		], dim=3).reshape(
			batch * roles,
			(1 + mask_guided_rgb.STATUS_DIM) * mask_guided_rgb.STACK_FRAMES,
			mask_guided_rgb.IMAGE_SIZE, mask_guided_rgb.IMAGE_SIZE,
		)
		role_feature = self.role_stem(role_input)
		role_feature = role_feature + self.status_projection(
			status.reshape(batch * roles, -1)
		).reshape(batch * roles, self.hidden_channels, 1, 1)
		role_feature = role_feature.reshape(
			batch, roles, self.hidden_channels, 4, 4
		)
		mask_feature = role_feature.amax(dim=1)
		film = self.film_output(mask_feature)
		scale, bias = film.chunk(2, dim=1)
		limit = self.correction_limit
		correction = limit * (
			spatial * torch.tanh(scale) + torch.tanh(bias)
		)
		if self.ablation_mode == 'guidance_off':
			# Keep the complete branch and parameterization matched, but make it
			# causally incapable of changing the official RGB representation.
			correction = correction * 0.0
		latent = self.rgb_norm(self.rgb_flatten(spatial + correction))
		return (latent, correction) if return_correction else latent


def enc(cfg, out=None):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	out = {} if out is None else out
	if mask_guided_rgb.enabled(cfg):
		mask_guided_rgb.validate_config(cfg)
		out['rgb'] = conv(
			cfg.obs_shape['rgb'], cfg.num_channels, act=SimNorm(cfg)
		)
		return nn.ModuleDict(out)
	if cfg.get('flat_anchor', False):
		if cfg.multitask:
			raise NotImplementedError('FlatAnchor does not support multitask training.')
		mode = cfg.get('flat_anchor_mode', 'residual')
		if mode not in {
			'residual', 'oc', 'object_graph', 'hybrid_graph', 'reward_graph',
			'cutie_hybrid', 'cutie_object_only',
		}:
			raise ValueError(f'Unknown flat_anchor_mode: {mode!r}.')
		if mode == 'cutie_object_only':
			if robust_object_field.enabled(cfg):
				robust_object_field.validate_config(cfg)
				# ROF owns a joint RGB/mask/descriptor encoder installed by
				# WorldModel after official module initialization. Constructing a
				# generic RGB encoder here would create an untracked bypass.
				return nn.ModuleDict(out)
			observation_variant = cfg.get(
				'cutie_object_observation_variant', 'full'
			)
			variable_graph = bool(cfg.get(
				'cutie_object_variable_graph_enabled', False
			))
			num_roles = int(cfg.get('cutie_object_num_roles', 2))
			expected_object_shape = (
				(2, 1774) if observation_variant == 'cutie_proprio'
				else (2, 25) if observation_variant == 'multimodal_articulated_pose'
				else (2, 21) if observation_variant in {
					'gt_articulated_pose', 'visual_articulated_pose'
				} else (num_roles, 1770)
			)
			if set(cfg.obs_shape) != {'object'}:
				raise ValueError(
					"CutieObjectOnly expects exactly one observation key: 'object'."
				)
			if tuple(cfg.obs_shape['object']) != expected_object_shape:
				raise ValueError(
					"CutieObjectOnly observation 'object' must have shape "
					f'{expected_object_shape} for {observation_variant}.'
				)
			if (
				not variable_graph
				and num_roles != fixed_cutie_role_count(cfg)
			):
				raise ValueError(
					'CutieObjectOnly role count is incompatible with its fixed MLP mode.'
				)
			if variable_graph and not 1 <= num_roles <= int(cfg.get(
				'cutie_object_variable_graph_max_roles', 8
			)):
				raise ValueError('Variable CutieObjectOnly role count is invalid.')
			if cfg.get('cutie_object_role_dim', 64) != 64:
				raise ValueError(
					'CutieObjectOnly requires 64 latent values per object role.'
				)
			expected_latent = (
				(
					num_roles * cfg.get('cutie_object_role_dim', 64)
					if cfg.get('cutie_object_variable_graph_readout', 'pool') == 'direct'
					else int(cfg.get('cutie_object_only_latent_dim', 128))
				)
				if variable_graph else
				num_roles * cfg.get('cutie_object_role_dim', 64)
			)
			if cfg.latent_dim != expected_latent or cfg.get(
				'cutie_object_only_latent_dim', 128
			) != expected_latent:
				raise ValueError(
					'CutieObjectOnly controller state width is inconsistent.'
				)
			# The live wrapper has already consumed RGB for perception.  Return an
			# empty container so no RGB or generic encoder is ever constructed.
			return nn.ModuleDict(out)
		if mode == 'cutie_hybrid':
			if 'rgb' not in cfg.obs_shape or 'object' not in cfg.obs_shape:
				raise ValueError(
					"CutieHybrid expects observation keys 'rgb' and 'object'."
				)
			if tuple(cfg.obs_shape['object']) != (2, 1770):
				raise ValueError(
					"CutieHybrid observation 'object' must have shape (2, 1770)."
				)
			rgb_latent_dim = 4 * 4 * cfg.num_channels
			if rgb_latent_dim != cfg.latent_dim:
				raise ValueError(
					'CutieHybrid must construct the official RGB path at its '
					f'original width ({rgb_latent_dim}), got {cfg.latent_dim}.'
				)
			if cfg.get('cutie_object_num_roles', 2) != 2:
				raise ValueError('CutieHybrid requires exactly two object roles.')
			if cfg.get('cutie_object_role_dim', 64) != 64:
				raise ValueError('CutieHybrid requires 64 latent values per object role.')
			if cfg.get('cutie_object_joint_dim', 640) != cfg.latent_dim + 128:
				raise ValueError('CutieHybrid joint state must be scene512+objects128=640.')
			# Construct the untouched official RGB encoder. The object modules are
			# installed only after every official module has been initialized.
			out['rgb'] = conv(
				cfg.obs_shape['rgb'], cfg.num_channels, act=SimNorm(cfg)
			)
			return nn.ModuleDict(out)
		if 'anchor' not in cfg.obs_shape:
			raise ValueError("FlatAnchor expects an observation key 'anchor'.")
		anchor_shape = cfg.obs_shape['anchor']
		if len(anchor_shape) != 1 or anchor_shape[0] != cfg.flat_anchor_dim:
			raise ValueError(
				f"Expected anchor observation shape ({cfg.flat_anchor_dim},), got {anchor_shape}."
			)

		if mode == 'object_graph':
			if cfg.flat_anchor_num_roles != 4:
				raise ValueError('ObjectGraph mode requires four ordered anchor roles.')
			role_dim = cfg.get('flat_anchor_object_role_dim', 32)
			if cfg.latent_dim != cfg.flat_anchor_num_roles * role_dim:
				raise ValueError('ObjectGraph latent_dim must equal four object role states.')
			# The branch is installed after official TD-MPC2 initialization. Return
			# an empty container here so no RGB CNN is ever constructed.
			return nn.ModuleDict(out)

		if 'rgb' not in cfg.obs_shape:
			raise ValueError(
				"Residual, OC, HybridGraph, and RewardGraph modes expect an observation key 'rgb'."
			)
		rgb_latent_dim = 4 * 4 * cfg.num_channels
		if mode in {'residual', 'hybrid_graph', 'reward_graph'} and rgb_latent_dim != cfg.latent_dim:
			raise ValueError(
				'Identity-preserving RGB/HybridGraph fusion requires the convolution output '
				f'({rgb_latent_dim}) to equal latent_dim ({cfg.latent_dim}).'
			)
		if mode == 'hybrid_graph':
			graph_dim = cfg.flat_anchor_num_roles * cfg.get('flat_anchor_object_role_dim', 32)
			if cfg.flat_anchor_num_roles != 4:
				raise ValueError('HybridGraph mode requires four ordered anchor roles.')
			if cfg.flat_anchor_scene_dim != cfg.latent_dim:
				raise ValueError('HybridGraph must construct the official scene path at its original width.')
			if cfg.get('flat_anchor_hybrid_joint_dim', cfg.flat_anchor_scene_dim + graph_dim) != cfg.flat_anchor_scene_dim + graph_dim:
				raise ValueError('HybridGraph joint_dim must equal scene_dim plus four role states.')
		if mode == 'reward_graph':
			graph_dim = cfg.flat_anchor_num_roles * cfg.get('flat_anchor_object_role_dim', 32)
			if cfg.flat_anchor_num_roles != 4:
				raise ValueError('RewardGraph mode requires four ordered anchor roles.')
			if cfg.flat_anchor_scene_dim != cfg.latent_dim:
				raise ValueError('RewardGraph must construct the official scene path at its original width.')
			if cfg.get('flat_anchor_hybrid_joint_dim', cfg.flat_anchor_scene_dim + graph_dim) != cfg.flat_anchor_scene_dim + graph_dim:
				raise ValueError('RewardGraph joint_dim must equal scene_dim plus four role states.')
		if mode == 'oc':
			if cfg.flat_anchor_num_roles != 4:
				raise ValueError('OC mode requires four ordered anchor roles.')
			if cfg.latent_dim != cfg.flat_anchor_joint_dim:
				raise ValueError('OC mode requires latent_dim to be expanded to joint_dim.')
			if cfg.flat_anchor_scene_dim + cfg.flat_anchor_num_roles * cfg.flat_anchor_role_dim != cfg.latent_dim:
				raise ValueError('OC scene and role dimensions must sum to latent_dim.')
			if cfg.flat_anchor_token_dim % cfg.flat_anchor_num_heads:
				raise ValueError('OC token dimension must be divisible by num_heads.')
		# Construct exactly the official RGB encoder here. WorldModel initializes
		# all official modules first. The selected anchor branch is installed only
		# after shared TD-MPC2 parameters have been initialized.
		out['rgb'] = conv(
			cfg.obs_shape['rgb'],
			cfg.num_channels,
			act=SimNorm(cfg),
		)
		return nn.ModuleDict(out)

	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict

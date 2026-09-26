"""Strict single-checkpoint evaluator for the Cutie cross-task pilot.

The evaluator reconstructs one training run from its runtime snapshot, switches
only the background split and the optional visual-only foreground erosion, and
records every 500-step episode. RGB, CutieHybrid, structural CutieObjectOnly,
and RobustObjectField
runs use identical environment/background/planner seed domains. By default only
``final.pt`` is accepted; ``--checkpoint-step`` explicitly selects a periodic
``eval_<step>.pt`` snapshot without weakening source/run identity validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))


FORMAT = 'cutie_multitask_checkpoint_evaluation_v1'
TASKS = (
	'reacher-visual-small',
	'cup-catch',
	'cartpole-swingup',
	'finger-spin',
	'acrobot-swingup',
	'reacher-easy', 'reacher-hard',
	'cartpole-balance', 'cartpole-balance-sparse', 'cartpole-swingup-sparse',
	'finger-turn-easy', 'finger-turn-hard', 'pendulum-swingup',
	'hopper-stand', 'hopper-hop',
	'walker-stand', 'walker-walk', 'walker-run',
	'cheetah-run', 'quadruped-run', 'quadruped-walk',
)
BACKENDS = (
	'rgb', 'cutie_masked_rgb', 'cutie_hybrid', 'cutie_object_only',
	'cutie_proprio_factorized', 'cutie_proprio_factorized_proprio_only',
	'robust_object_field', 'cutie_mask_guided_rgb',
)
CONDITIONS = ('clean', 'hard')
VALIDATION_SPLIT = 'validation'
BACKGROUND_SPLITS = ('train', 'validation', 'test')
LEGACY_MLP_K3_V1 = 'legacy_mlp_k3_v1'


def _legacy_mlp_k3_latent(raw: dict, backend: str):
	"""Return 192 only for the exact non-spatial three-role legacy mode.

	The established two-role evaluator path intentionally remains unchanged.
	Keeping this check local to the evaluator prevents a 100k K=3 training run
	from being rejected only when its held-out clean/hard evaluation begins.
	"""
	if backend != 'cutie_object_only' or raw.get(
		'cutie_object_regression_encoder'
	) != LEGACY_MLP_K3_V1:
		return None
	expected = {
		'cutie_object_num_roles': 3,
		'cutie_object_role_dim': 64,
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_only_latent_dim': 192,
		'cutie_object_observation_variant': 'full',
		'cutie_object_spatial_token_enabled': False,
		'cutie_object_variable_graph_enabled': False,
	}
	bad = {
		key: (raw.get(key), value)
		for key, value in expected.items()
		if raw.get(key) != value
	}
	if bad:
		raise ValueError(f'K3 legacy evaluator contract mismatch: {bad}.')
	return 192


def _robust_object_field_latent(raw: dict, backend: str, erosion_pixels=0):
	"""Validate the actual ROF source instead of relabeling a legacy run."""
	if backend != 'robust_object_field':
		if raw.get('robust_object_field_enabled', False):
			raise ValueError('ROF source requires its explicit evaluator backend.')
		return None
	from common import robust_object_field as rof
	from envs.wrappers.robust_object_field import (
		validate_robust_object_field_observation_config,
	)
	validate_robust_object_field_observation_config(raw)
	# A runtime snapshot must retain the complete four-array observation schema.
	rof.validate_config(raw)
	if int(erosion_pixels) != 0:
		raise ValueError('ROF V0 held-out evaluation does not support foreground erosion.')
	if int(raw.get('robust_object_field_random_shift_pad', 3)) != 3:
		raise ValueError('Paired ROF evaluation requires the native pad-three RGB shift.')
	return rof.latent_dim(raw)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file:
		for block in iter(lambda: file.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _json(path: Path) -> dict:
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, dict):
		raise ValueError(f'Expected JSON object: {path}')
	return value


def _walk_env(env):
	seen = set()
	while env is not None and id(env) not in seen:
		seen.add(id(env))
		yield env
		env = getattr(env, 'env', None)


def _declares(value, name):
	return any(name in cls.__dict__ for cls in type(value).__mro__)


def _env_value(env, name, default=None):
	for value in _walk_env(env):
		if _declares(value, name):
			return getattr(value, name)
	return default


def _metrics(env):
	for value in _walk_env(env):
		if _declares(value, 'metrics') and callable(value.metrics):
			result = value.metrics()
			return dict(result) if isinstance(result, dict) else None
	return None


def _tensor_hash(value) -> str:
	array = value.detach().cpu().contiguous().numpy()
	return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def _rgb_hash(obs, env, backend: str) -> str:
	if backend == 'rgb':
		value = obs.detach().cpu()
		if tuple(value.shape) != (9, 64, 64) or str(value.dtype) != 'torch.uint8':
			raise RuntimeError(
				f'RGB baseline observation must be uint8 [9,64,64], got '
				f'{tuple(value.shape)} {value.dtype}.'
			)
		# Match CutieObjectWrapper.latest_source_rgb_sha256 exactly: newest
		# native CHW frame converted to contiguous HWC bytes.
		array = value[-3:].permute(1, 2, 0).contiguous().numpy()
		return hashlib.sha256(array.tobytes(order='C')).hexdigest()
	value = _env_value(env, 'latest_source_rgb_sha256')
	if not isinstance(value, str) or len(value) != 64:
		raise RuntimeError('Cutie wrapper did not expose the source RGB hash.')
	if backend in {'robust_object_field', 'cutie_mask_guided_rgb'}:
		array = obs['rgb'][-3:].detach().cpu().permute(1, 2, 0).contiguous().numpy()
		if hashlib.sha256(array.tobytes(order='C')).hexdigest() != value:
			raise RuntimeError(
				f'{backend} observation RGB differs from the Cutie source frame.'
			)
	return value


def _object_hash(obs, backend: str):
	if backend in {'rgb', 'cutie_masked_rgb'}:
		try:
			keys = set(obs.keys())
		except Exception:
			return None
		if keys:
			raise RuntimeError(f'RGB baseline unexpectedly returned keys {keys}.')
		return None
	keys = set(obs.keys())
	expected = (
		{'rgb', 'object', 'object_mask', 'role_exists'}
		if backend == 'robust_object_field'
		else {'rgb', 'object_mask', 'tracker_status'}
		if backend == 'cutie_mask_guided_rgb'
		else {'rgb', 'object'} if backend == 'cutie_hybrid' else {'object'}
	)
	if keys != expected:
		raise RuntimeError(f'{backend} observation keys {keys} != {expected}.')
	if backend == 'robust_object_field':
		from common.robust_object_field_observation import validate_numpy_observation
		arrays = {
			key: value.detach().cpu().contiguous().numpy()
			for key, value in obs.items()
		}
		validate_numpy_observation(arrays, role_count=len(arrays['role_exists']))
	if backend == 'cutie_mask_guided_rgb':
		if obs['rgb'].dtype != torch.uint8 or obs['object_mask'].dtype != torch.bool:
			raise RuntimeError('Mask-guided RGB observation dtypes are invalid.')
		if obs['tracker_status'].dtype != torch.float32:
			raise RuntimeError('Mask-guided tracker status must be float32.')
		if not bool(torch.isfinite(obs['tracker_status']).all().item()):
			raise RuntimeError('Mask-guided tracker status is non-finite.')
		return None
	return _tensor_hash(obs['object'])


def _mask_guided_binding_hash(obs, source_rgb_sha256: str) -> str:
	"""Recompute the wrapper's exact same-response public tensor binding."""
	digest = hashlib.sha256()
	digest.update(source_rgb_sha256.encode('ascii'))
	for key in ('object_mask', 'tracker_status'):
		array = obs[key].detach().cpu().contiguous().numpy()
		digest.update(array.tobytes(order='C'))
	return digest.hexdigest()


def _mask_guided_runtime(raw, perception, ready):
	from common import mask_guided_rgb as guided
	if not isinstance(perception, dict) or not isinstance(ready, dict):
		raise RuntimeError('Mask-guided RGB runtime provenance is unavailable.')
	roles = list(raw['cutie_object_role_names'])
	expected_ready = {
		'treatment': guided.SCHEMA,
		'policy_observation': guided.SCHEMA,
		'policy_original_rgb_background': True,
		'policy_native_role_masks': True,
		'policy_tracker_status': True,
		'policy_cutie_descriptors': False,
		'role_names': roles,
		'role_count': len(roles),
		'role_axis': 'task_static_exact_k_no_padding',
		'rgb_mask_status_binding': (
			'same_synchronous_cutie_response_native64_three_frame_causal_v1'
		),
		'privileged_runtime_segmentation': False,
		'privileged_runtime_kinematics': False,
	}
	for key, expected in expected_ready.items():
		if ready.get(key) != expected:
			raise RuntimeError(
				f'Mask-guided ready {key}={ready.get(key)!r}, expected {expected!r}.'
			)
	if perception.get('mask_guided_rgb_schema') != guided.SCHEMA:
		raise RuntimeError('Mask-guided runtime schema mismatch.')
	if perception.get('mask_guided_rgb_frames') != 20 * 501:
		raise RuntimeError('Mask-guided public observation frame count mismatch.')
	if perception.get('mask_guided_rgb_role_count') != len(roles):
		raise RuntimeError('Mask-guided runtime role count mismatch.')
	value = perception.get('mask_guided_rgb_last_binding_sha256')
	if not isinstance(value, str) or len(value) != 64:
		raise RuntimeError('Mask-guided final binding hash is unavailable.')


def _robust_object_field_runtime(raw, perception, ready):
	"""Require evidence that the four-array wrapper actually ran every frame."""
	from common.robust_object_field_observation import SCHEMA
	roles = list(raw['cutie_object_role_names'])
	if not isinstance(perception, dict) or not isinstance(ready, dict):
		raise RuntimeError('ROF runtime provenance is unavailable.')
	checks = {
		'treatment': SCHEMA,
		'policy_observation': SCHEMA,
		'policy_rgb': True,
		'policy_cutie_descriptors': True,
		'policy_native_role_masks': True,
		'role_names': roles,
		'role_count': len(roles),
		'role_axis': 'task_static_exact_k_no_padding',
		'role_exists': 'all_one',
		'rgb_descriptor_mask_binding': (
			'same_synchronous_cutie_response_native64_three_frame_causal_v1'
		),
		'privileged_runtime_segmentation': False,
		'privileged_runtime_kinematics': False,
	}
	for key, expected in checks.items():
		if ready.get(key) != expected:
			raise RuntimeError(f'ROF ready {key}={ready.get(key)!r}, expected {expected!r}.')
	for key, expected in {
		'robust_object_field_schema': SCHEMA,
		'robust_object_field_frames': 20 * 501,
		'robust_object_field_role_count': len(roles),
		'robust_object_field_last_sequence_id': 20 * 501 - 1,
	}.items():
		if perception.get(key) != expected:
			raise RuntimeError(
				f'ROF metric {key}={perception.get(key)!r}, expected {expected!r}.'
			)
	for kind in ('source_rgb', 'object', 'mask', 'binding'):
		value = perception.get(f'robust_object_field_last_{kind}_sha256')
		if (
			not isinstance(value, str) or len(value) != 64
			or any(char not in '0123456789abcdef' for char in value)
		):
			raise RuntimeError(f'ROF {kind} provenance hash is missing or malformed.')


def _validate_training_condition(raw, condition):
	if condition not in CONDITIONS:
		raise ValueError(f'Unknown training condition: {condition!r}.')
	if raw.get('video_background_enabled') is not (condition == 'hard'):
		raise ValueError(f'Source background enabled does not match {condition} training.')
	if raw.get('video_background_split') != 'train':
		raise ValueError('Source config must retain the frozen train video split.')


def _evaluation_background(condition, requested_split=None):
	"""Resolve the visual evaluation domain without changing legacy callers.

	Historically ``condition=hard`` always meant the validation background
	split.  New publication runners need to distinguish a seen train split from
	a genuinely held-out test split, while old commands and archived provenance
	must keep their original meaning.  Therefore an omitted split retains the
	legacy validation behavior.
	"""
	if condition not in CONDITIONS:
		raise ValueError(f'Unknown evaluation condition: {condition!r}.')
	if condition == 'clean':
		if requested_split not in (None, 'clean'):
			raise ValueError('Clean evaluation cannot select a video background split.')
		return False, 'clean'
	resolved = VALIDATION_SPLIT if requested_split is None else str(requested_split)
	if resolved not in BACKGROUND_SPLITS:
		raise ValueError(
			f'Hard evaluation background split must be one of '
			f'{BACKGROUND_SPLITS!r}, got {resolved!r}.'
		)
	return True, resolved


def _rng_hash(torch) -> str:
	digest = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes())
	digest.update(torch.cuda.get_rng_state(0).cpu().numpy().tobytes())
	return digest.hexdigest()


def _align_object_only_rgb_shift_rng(torch, backend: str) -> None:
	"""Keep MPPI randomness paired with the two RGB-encoding backends.

	The official RGB encoder begins with ``ShiftAug(pad=3)`` and consumes one
	CUDA ``randint`` tensor of shape ``[B,1,1,2]`` on every call. Structural
	object-only deliberately has no RGB encoder, so without this no-op draw its
	MPPI samples would start at a different point in the RNG stream even when the
	episode planner seed is identical. Evaluation is batch-one and frozen to the
	official pad of three.
	"""
	if backend in {
		'cutie_object_only', 'cutie_proprio_factorized',
		'cutie_proprio_factorized_proprio_only',
	}:
		torch.randint(
			0, 7, size=(1, 1, 1, 2), device='cuda:0', dtype=torch.float32
		)


def _prepare(args, raw: dict):
	from common import MODEL_SIZE
	from common.parser import cfg_to_dataclass
	from omegaconf import OmegaConf

	if args.task not in TASKS or raw.get('task') != args.task:
		raise ValueError((args.task, raw.get('task')))
	if raw.get('obs') != 'rgb' or raw.get('multitask') is not False:
		raise ValueError('Cross-task pilot requires single-task RGB source runs.')
	if raw.get('model_size') != 5 or int(raw.get('seed', -1)) != args.training_seed:
		raise ValueError('Source model-size/seed mismatch.')
	for key, expected in {
		'steps': args.expected_training_steps,
		'eval_freq': args.expected_training_eval_freq,
		'eval_episodes': args.expected_training_eval_episodes,
		'episode_length': 500,
	}.items():
		if int(raw.get(key, -1)) != int(expected):
			raise ValueError(f'Source {key}={raw.get(key)!r}, expected {expected}.')
	_validate_training_condition(raw, getattr(args, 'training_condition', 'hard'))
	field_latent = _robust_object_field_latent(raw, args.backend, args.erosion_pixels)
	if args.backend == 'cutie_mask_guided_rgb':
		from common import mask_guided_rgb as guided
		from envs.wrappers.cutie_mask_guided_rgb import (
			validate_mask_guided_rgb_observation_config,
		)
		validate_mask_guided_rgb_observation_config(raw)
		guided.validate_config(raw)
	elif raw.get('cutie_mask_guided_rgb_enabled', False):
		raise ValueError(
			'Mask-guided RGB source requires its explicit evaluator backend.'
		)
	if args.backend == 'cutie_masked_rgb':
		from envs.wrappers.cutie_masked_rgb import validate_masked_rgb_config
		validate_masked_rgb_config(raw)
	elif raw.get('cutie_masked_rgb_enabled', False):
		raise ValueError('Masked RGB source requires its explicit evaluator backend.')
	if int(raw.get('visual_foreground_erosion_pixels', -1)) != 0:
		raise ValueError('Source training must use zero foreground erosion.')
	for key, expected in {
		'enc_dim': 256,
		'mlp_dim': 512,
		'num_enc_layers': 2,
	}.items():
		if int(raw.get(key, -1)) != expected:
			raise ValueError(f'Source MODEL_SIZE[5] field {key} is not {expected}.')

	expected_mode = {
		'rgb': (False, None, 512),
		'cutie_masked_rgb': (False, None, 512),
		'cutie_mask_guided_rgb': (False, None, 512),
		'cutie_hybrid': (True, 'cutie_hybrid', 640),
		'cutie_object_only': (True, 'cutie_object_only', 128),
		'cutie_proprio_factorized': (True, 'cutie_object_only', 128),
		'cutie_proprio_factorized_proprio_only': (
			True, 'cutie_object_only', 128,
		),
		'robust_object_field': (True, 'cutie_object_only', field_latent),
	}[args.backend]
	flat, mode, latent = expected_mode
	k3_latent = _legacy_mlp_k3_latent(raw, args.backend)
	if k3_latent is not None:
		latent = k3_latent
	elif (
		args.backend == 'cutie_object_only'
		and raw.get('cutie_object_variable_graph_enabled') is True
		and raw.get('cutie_object_variable_graph_readout') == 'direct'
	):
		num_roles = int(raw.get('cutie_object_num_roles', -1))
		role_dim = int(raw.get('cutie_object_role_dim', -1))
		if num_roles < 1 or role_dim < 1:
			raise ValueError(
				'Variable direct object graph requires positive role count and role dim.'
			)
		latent = num_roles * role_dim
		if int(raw.get('cutie_object_only_latent_dim', -1)) != latent:
			raise ValueError(
				'Variable direct object graph latent contract mismatch: '
				f'{raw.get("cutie_object_only_latent_dim")} != {num_roles}*{role_dim}.'
			)
	if bool(raw.get('flat_anchor')) is not flat:
		raise ValueError(f'{args.backend} flat_anchor mismatch.')
	if flat and raw.get('flat_anchor_mode') != mode:
		raise ValueError(f'{args.backend} mode mismatch.')
	if int(raw.get('latent_dim', -1)) != latent:
		raise ValueError(f'{args.backend} latent_dim mismatch: {raw.get("latent_dim")}.')
	if args.backend in {
		'cutie_proprio_factorized', 'cutie_proprio_factorized_proprio_only',
	}:
		expected_cutie_proprio = {
			'cutie_object_observation_variant': 'cutie_proprio',
			'cutie_proprio_mode': (
				'factorized'
				if args.backend == 'cutie_proprio_factorized'
				else 'factorized_proprio_only'
			),
			'cutie_object_frame_dim': 1774,
			'cutie_object_input_dim': 1774,
			'cutie_object_only_latent_dim': 128,
			'cutie_object_allow_simulator_runtime': False,
			'cutie_object_allow_simulator_kinematics_runtime': True,
		}
		bad = {
			key: (raw.get(key), value)
			for key, value in expected_cutie_proprio.items()
			if raw.get(key) != value
		}
		if bad:
			raise ValueError(f'Cutie-proprio runtime contract mismatch: {bad}.')

	data = dict(raw)
	data.update(MODEL_SIZE[5])
	# MODEL_SIZE[5] restores the official 512-D default.  Object-only K=3 has no
	# RGB scene encoder to preserve, so retain the checkpoint's explicit 192-D
	# controller width before environment and agent reconstruction.  All other
	# evaluator modes keep their historical initialization path bit-for-bit.
	if k3_latent is not None:
		data['latent_dim'] = k3_latent
	if field_latent is not None:
		# Preserve the exact K*5*64 checkpoint width after MODEL_SIZE's 512-D
		# default. ROF validation must also remain valid after reconstruction.
		data['latent_dim'] = field_latent
		data['cutie_object_only_latent_dim'] = field_latent
	# The train-only state decoder stays in the checkpoint architecture, but
	# the evaluation environment must not read any privileged state labels.
	data['object_state_supervision_collect_labels'] = False
	background_enabled, background_split = _evaluation_background(
		args.condition, getattr(args, 'background_split', None),
	)
	data.update(
		seed=int(args.env_seed),
		video_background_seed=int(args.background_seed),
		video_background_split=(
			background_split if background_enabled else raw['video_background_split']
		),
		video_background_enabled=background_enabled,
		visual_foreground_erosion_pixels=int(args.erosion_pixels),
		eval_episodes=int(args.episodes),
		compile=False,
		compile_fallback_random=False,
		save_video=False,
		save_csv=False,
		save_agent=False,
		enable_wandb=False,
		checkpoint=str(args.checkpoint.resolve()),
		exp_name=f'multitask_eval_{args.backend}',
		work_dir=str(args.output.parent),
	)
	cfg = cfg_to_dataclass(OmegaConf.create(data))
	cfg.work_dir = args.output.parent
	return cfg


def _validate_args(args):
	if args.episodes != 20:
		raise ValueError('Frozen pilot evaluation requires exactly 20 episodes.')
	if args.erosion_pixels not in (0, 1, 2):
		raise ValueError('--erosion-pixels must be 0, 1, or 2.')
	if args.condition == 'clean' and args.erosion_pixels != 0:
		raise ValueError('Clean evaluation does not support foreground erosion.')
	_evaluation_background(
		args.condition, getattr(args, 'background_split', None),
	)
	if len({args.env_seed, args.background_seed, args.planner_seed_base}) != 3:
		raise ValueError('Environment/background/planner seed domains must differ.')
	if args.output.exists():
		raise FileExistsError(args.output)
	for path in (args.runtime_config, args.checkpoint):
		if not path.is_file():
			raise FileNotFoundError(path)
	if args.checkpoint_step is None:
		expected = args.runtime_config.resolve().parent / 'models' / 'final.pt'
	else:
		if args.checkpoint_step <= 0:
			raise ValueError('--checkpoint-step must be positive.')
		if args.checkpoint_step > args.expected_training_steps:
			raise ValueError('--checkpoint-step exceeds the training budget.')
		if args.checkpoint_step % args.expected_training_eval_freq != 0:
			raise ValueError(
				'--checkpoint-step must coincide with a periodic evaluation.'
			)
		expected = (
			args.runtime_config.resolve().parent / 'models'
			/ f'eval_{args.checkpoint_step}.pt'
		)
	if args.checkpoint.resolve() != expected:
		raise ValueError(f'Checkpoint/runtime mismatch: {args.checkpoint} != {expected}')


def evaluate(args):
	import numpy as np
	import torch
	from common.seed import set_seed
	from envs import make_env
	from isi_wm import TDMPC2
	from common import object_state_supervision

	_validate_args(args)
	raw = _json(args.runtime_config)
	cfg = _prepare(args, raw)
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required.')
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')

	env = None
	records = []
	object_only_alignment_draws = 0
	actual_erosion = None
	state_label_reads = None
	measure_latency = bool(getattr(args, 'measure_online_latency', False))
	latencies = {'agent_act': [], 'env_step': [], 'total': []}
	started = perf_counter()
	try:
		set_seed(args.env_seed)
		env = make_env(cfg)
		active_split = _env_value(env, 'active_split')
		background_enabled, expected_background_split = _evaluation_background(
			args.condition, getattr(args, 'background_split', None),
		)
		if background_enabled and active_split != expected_background_split:
			raise RuntimeError(
				f'{expected_background_split!r} background split was not constructed.'
			)
		if args.condition == 'clean' and active_split is not None:
			raise RuntimeError(
				'Clean evaluation unexpectedly constructed a background wrapper.'
			)
		actual_erosion = _env_value(env, 'erosion_pixels')
		if args.erosion_pixels == 0:
			if actual_erosion not in (None, 0):
				raise RuntimeError(
					f'Unexpected foreground erosion wrapper radius {actual_erosion}.'
				)
		elif actual_erosion != args.erosion_pixels:
			raise RuntimeError(
				'Foreground erosion wrapper is missing or has the wrong radius: '
				f'expected={args.erosion_pixels}, actual={actual_erosion!r}.'
			)
		# Environment and model seed domains are deliberately independent.  The
		# evaluation environment uses ``env_seed`` above, while checkpoint
		# contracts (including the deterministic auxiliary pairing transcript)
		# were bound to the original training seed.  Reconstruct the agent with
		# that training seed so evaluation cannot spuriously reject an otherwise
		# valid checkpoint merely because the held-out environment seed differs.
		agent_cfg = copy.deepcopy(cfg)
		agent_cfg.seed = int(args.training_seed)
		agent = TDMPC2(agent_cfg)
		agent.load(args.checkpoint)
		agent.eval()
		for episode_index in range(args.episodes):
			obs = env.reset()
			planner_seed = args.planner_seed_base + episode_index
			initial_rgb = _rgb_hash(obs, env, args.backend)
			initial_object = _object_hash(obs, args.backend)
			initial_field = {}
			if args.backend == 'robust_object_field':
				binding = _env_value(env, 'robust_object_field_last_binding_sha256')
				if not isinstance(binding, str) or len(binding) != 64:
					raise RuntimeError('ROF reset observation binding is unavailable.')
				initial_field = {
					'initial_object_mask_sha256': _tensor_hash(obs['object_mask']),
					'initial_role_exists_sha256': _tensor_hash(obs['role_exists']),
					'initial_field_binding_sha256': binding,
				}
			elif args.backend == 'cutie_mask_guided_rgb':
				binding = _env_value(
					env, 'mask_guided_rgb_last_binding_sha256'
				)
				expected_binding = _mask_guided_binding_hash(obs, initial_rgb)
				if binding != expected_binding:
					raise RuntimeError(
						'Mask-guided reset observation binding is unavailable or changed.'
					)
				initial_field = {
					'initial_object_mask_sha256': _tensor_hash(obs['object_mask']),
					'initial_tracker_status_sha256': _tensor_hash(
						obs['tracker_status']
					),
					'initial_mask_guided_binding_sha256': binding,
				}
			source = _env_value(env, 'active_source')
			frame_index = _env_value(env, 'frame_index')
			if background_enabled and (source is None or frame_index is None):
				raise RuntimeError('Background provenance is unavailable.')
			if args.condition == 'clean':
				source, frame_index = 'clean', 0
			set_seed(planner_seed)
			agent._prev_mean.zero_()
			rng_start = _rng_hash(torch)
			reward_sum = 0.0
			info = {}
			for step_index in range(int(cfg.episode_length)):
				torch.compiler.cudagraph_mark_step_begin()
				_align_object_only_rgb_shift_rng(torch, args.backend)
				object_only_alignment_draws += int(
					args.backend in {
						'cutie_object_only', 'cutie_proprio_factorized',
					}
				)
				if measure_latency:
					torch.cuda.synchronize()
					act_started = perf_counter()
				action = agent.act(obs, t0=step_index == 0, eval_mode=True)
				if measure_latency:
					torch.cuda.synchronize()
					act_ended = perf_counter()
				obs, reward, done, info = env.step(action)
				if measure_latency:
					torch.cuda.synchronize()
					step_ended = perf_counter()
					latencies['agent_act'].append(1000.0 * (act_ended - act_started))
					latencies['env_step'].append(1000.0 * (step_ended - act_ended))
					latencies['total'].append(1000.0 * (step_ended - act_started))
				reward_sum += float(reward)
				if done:
					length = step_index + 1
					break
			else:
				raise RuntimeError('Environment did not terminate at 500 steps.')
			if length != 500:
				raise RuntimeError(f'Unexpected episode length {length}.')
			records.append({
				'episode_index': episode_index,
				'planner_seed': planner_seed,
				'planner_rng_start_sha256': rng_start,
				'planner_rng_end_sha256': _rng_hash(torch),
				'initial_rgb_sha256': initial_rgb,
				'initial_object_sha256': initial_object,
				'background_source': Path(source).name,
				'background_start_frame_index': int(frame_index),
				'reward': reward_sum,
				'success': float(info.get('success', 0.0)),
				'length': length,
				**initial_field,
			})
			print('CUTIE_MULTITASK_EVAL_EPISODE', json.dumps({
				'task': args.task, 'backend': args.backend,
				'condition': args.condition,
				'erosion_pixels': args.erosion_pixels, **records[-1],
			}, allow_nan=False), flush=True)
		perception = _metrics(env)
		ready = _env_value(env, 'cutie_ready')
		manifest = _env_value(env, 'manifest_sha256')
		combined_manifest = _env_value(env, 'combined_manifest_sha256')
		if raw.get('object_state_supervision_enabled', False):
			state_label_reads = _env_value(env, 'state_supervision_label_reads')
			if state_label_reads != 0:
				raise RuntimeError(f'Evaluation read privileged state labels: {state_label_reads!r}.')
	finally:
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()

	rewards = np.asarray([record['reward'] for record in records], dtype=np.float64)
	if rewards.size != 20 or not np.isfinite(rewards).all():
		raise RuntimeError('Incomplete/non-finite evaluation rewards.')
	if args.backend == 'rgb':
		if perception is not None or ready is not None:
			raise RuntimeError('RGB baseline unexpectedly started Cutie.')
	else:
		if not isinstance(perception, dict) or not isinstance(ready, dict):
			raise RuntimeError('Cutie runtime provenance is unavailable.')
		if perception.get('frames') != 20 * 501:
			raise RuntimeError(f'Cutie frame count mismatch: {perception.get("frames")}')
	field_contract = None
	if args.backend == 'robust_object_field':
		from common import robust_object_field as rof
		_robust_object_field_runtime(raw, perception, ready)
		field_contract = rof.contract(cfg)
	mask_guided_contract = None
	if args.backend == 'cutie_mask_guided_rgb':
		from common import mask_guided_rgb as guided
		_mask_guided_runtime(raw, perception, ready)
		mask_guided_contract = dict(guided.contract(cfg))

	cutie_inputs = None
	if args.backend != 'rgb':
		checkpoint = Path(raw['cutie_object_checkpoint']).resolve()
		support = Path(raw['cutie_object_support_path']).resolve()
		cutie_inputs = {
			'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha256(checkpoint),
			'support': str(support), 'support_sha256': _sha256(support),
			'roles': raw.get('cutie_object_role_names'),
			'support_schema': raw.get('cutie_object_support_schema'),
		}
	return {
		'format': FORMAT,
		'protocol': {
			'training_condition': getattr(args, 'training_condition', 'hard'),
			'masked_rgb_enabled': bool(raw.get('cutie_masked_rgb_enabled', False)),
			'object_state_supervision': (
				object_state_supervision.contract(cfg)
				if raw.get('object_state_supervision_enabled', False) else None
			),
			'state_supervision_collect_labels': False,
			'state_supervision_label_reads': state_label_reads,
			'robust_object_field': field_contract,
			'mask_guided_rgb': mask_guided_contract,
		},
		'task': args.task,
		'backend': args.backend,
		'condition': args.condition,
		'training_seed': args.training_seed,
		'erosion_pixels': args.erosion_pixels,
		'evaluation': {
			'split': expected_background_split,
			'condition': args.condition, 'episodes': 20,
			'env_seed': args.env_seed, 'background_seed': args.background_seed,
			'planner_seed_base': args.planner_seed_base,
			'eval_mode': True,
			'rgb_shift_rng_alignment': (
				'rof_native_joint_shift_seed_domains_only_v1'
				if args.backend == 'robust_object_field'
				else 'mask_guided_native_joint_shift_v1'
				if args.backend == 'cutie_mask_guided_rgb'
				else 'object_only_equivalent_cuda_randint_v1'
			),
			'object_only_alignment_draws': object_only_alignment_draws,
			'expected_rgb_shift_draws_per_backend': 20 * 500,
			'actual_foreground_erosion_pixels': (
				0 if actual_erosion is None else int(actual_erosion)
			),
		},
		'provenance': {
			'runtime_config': str(args.runtime_config.resolve()),
			'runtime_config_sha256': _sha256(args.runtime_config),
			'checkpoint': str(args.checkpoint.resolve()),
			'checkpoint_sha256': _sha256(args.checkpoint),
			'checkpoint_kind': (
				'final' if args.checkpoint_step is None else 'periodic_eval'
			),
			'checkpoint_step': (
				int(args.expected_training_steps)
				if args.checkpoint_step is None else int(args.checkpoint_step)
			),
			'evaluator_sha256': _sha256(Path(__file__).resolve()),
			'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
			'device_name': torch.cuda.get_device_name(0),
			# Retain the legacy key for archived consumers while exposing the
			# split-neutral identity needed by train/test background evaluation.
			'validation_manifest_sha256': manifest,
			'background_manifest_sha256': manifest,
			'combined_manifest_sha256': combined_manifest,
			'cutie_inputs': cutie_inputs,
			'cutie_ready': ready,
		},
		'episodes': records,
		'summary': {
			'reward_mean': float(rewards.mean()),
			'reward_median': float(np.median(rewards)),
			'reward_std': float(rewards.std(ddof=1)),
			'reward_var': float(rewards.var(ddof=1)),
			'reward_min': float(rewards.min()),
			'reward_max': float(rewards.max()),
			'elapsed_seconds': float(perf_counter() - started),
		},
		'perception_runtime': perception,
		'online_latency': ({
			'semantics': 'synchronized_batch_one_agent_act_plus_env_step_including_render_cutie_masking;excludes_reset_load_logging',
			'samples': len(latencies['total']),
			'components': {
				name: {
					'mean_ms': float(np.mean(values)),
					'median_ms': float(np.median(values)),
					'p95_ms': float(np.percentile(values, 95)),
					'max_ms': float(np.max(values)),
				}
				for name, values in latencies.items()
			},
		} if measure_latency else None),
	}


def _write(path: Path, payload):
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + '.incomplete')
	if temporary.exists():
		raise FileExistsError(temporary)
	try:
		with temporary.open('x', encoding='utf-8', newline='\n') as file:
			json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
			file.write('\n')
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--backend', choices=BACKENDS, required=True)
	parser.add_argument('--condition', choices=CONDITIONS, default='hard')
	parser.add_argument(
		'--background-split', choices=BACKGROUND_SPLITS,
		help=(
			'Video split used when condition=hard. Omission preserves the legacy '
			'validation split; clean evaluation forbids this option.'
		),
	)
	parser.add_argument('--training-condition', choices=CONDITIONS, default='hard')
	parser.add_argument('--measure-online-latency', action='store_true')
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--checkpoint-step', type=int)
	parser.add_argument('--training-seed', type=int, default=6)
	parser.add_argument('--expected-training-steps', type=int, default=100000)
	parser.add_argument('--expected-training-eval-freq', type=int, default=20000)
	parser.add_argument('--expected-training-eval-episodes', type=int, default=3)
	parser.add_argument('--episodes', type=int, default=20)
	parser.add_argument('--env-seed', type=int, default=424243)
	parser.add_argument('--background-seed', type=int, default=1618034)
	parser.add_argument('--planner-seed-base', type=int, default=8675400)
	parser.add_argument('--erosion-pixels', type=int, default=0)
	parser.add_argument('--output', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	payload = evaluate(args)
	_write(args.output, payload)
	print('CUTIE_MULTITASK_EVAL_OK', json.dumps({
		'task': args.task, 'backend': args.backend,
		'condition': args.condition,
		'erosion_pixels': args.erosion_pixels,
		'reward_mean': payload['summary']['reward_mean'],
		'output': str(args.output.resolve()),
	}, allow_nan=False))
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

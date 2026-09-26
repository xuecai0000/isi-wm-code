"""Collect strict same-state real-action branches for ROF dynamics diagnosis.

This collector closes the central causal gap in observational ROF rollouts.  A
root state is reached by replaying one frozen, checkpoint-generated prefix from
a fresh deterministic reset.  Every sibling then starts from another fresh
reset, replays the exact same prefix, executes one member of a balanced real
action design (zero and +/- every action-space basis), and receives the same
open-loop continuation actions.  No future is paired with an action that was
not actually executed in the simulator.

Clean and hard observations are also produced by independent fresh-reset
replays.  Their physics, official task observations, GT masks, and executed
actions must match exactly before one shared oracle trace is published.  The
model-facing arrays contain only causal ROF observations and executed actions;
simulator state and segmentation live exclusively under ``oracle__`` keys.

The complete sibling set and both background twins are stored in one NPZ and
assigned to one root-level split.  This prevents branch or background twins
from leaking across train/validation/test.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for _local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while _local_path in sys.path:
		sys.path.remove(_local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))


FORMAT = 'rof_same_state_action_branch_dataset_v1'
MANIFEST_NAME = 'dataset_manifest.json'
TASKS = ('acrobot-swingup', 'cartpole-balance-sparse')
CONDITIONS = ('clean', 'hard')
HISTORY_FRAMES = 3
HORIZON = 5
POLICY_FIELDS = ('rgb', 'object', 'object_mask', 'role_exists')
POLICY_KEYS = tuple(
	f'{condition}__{part}__policy_{field}'
	for condition in CONDITIONS
	for part in ('history', 'future')
	for field in POLICY_FIELDS
)
ACTION_KEYS = ('history__action', 'branch__action')
BRANCH_KEYS = ('branch__code', 'branch__is_zero')
ORACLE_KEYS = (
	'oracle__official_state',
	'oracle__physics_state',
	'oracle__gt_role_mask',
	'oracle__gt_visible',
	'oracle__absolute_step',
	'oracle__hard_official_state',
	'oracle__hard_gt_role_mask',
	'oracle__clean_prefix_action',
	'oracle__hard_prefix_action',
	'oracle__clean_full_physics_state',
	'oracle__hard_full_physics_state',
)
GROUP_KEYS = frozenset(POLICY_KEYS + ACTION_KEYS + BRANCH_KEYS + ORACLE_KEYS)
MODEL_INPUT_KEYS = POLICY_KEYS + ACTION_KEYS
ORACLE_NAMESPACE = 'oracle__'
BRANCH_ACTION_INDEX = HISTORY_FRAMES - 1


def canonical_json_bytes(payload: Any) -> bytes:
	return (
		json.dumps(
			payload, ensure_ascii=False, sort_keys=True,
			separators=(',', ':'), allow_nan=False,
		) + '\n'
	).encode('utf-8')


def file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with Path(path).open('rb') as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def typed_array_sha256(name: str, value: np.ndarray) -> str:
	array = np.ascontiguousarray(np.asarray(value))
	digest = hashlib.sha256()
	digest.update(name.encode('utf-8'))
	digest.update(b'\0')
	digest.update(array.dtype.str.encode('ascii'))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes(order='C'))
	return digest.hexdigest()


def array_trace_sha256(
	arrays: Mapping[str, np.ndarray], keys: Sequence[str],
) -> str:
	digest = hashlib.sha256()
	for key in keys:
		if key not in arrays:
			raise ValueError(f'Array trace is missing {key!r}.')
		digest.update(key.encode('utf-8'))
		digest.update(typed_array_sha256(key, arrays[key]).encode('ascii'))
	return digest.hexdigest()


def model_input_trace_sha256(arrays: Mapping[str, np.ndarray]) -> str:
	return array_trace_sha256(arrays, MODEL_INPUT_KEYS)


def oracle_trace_sha256(arrays: Mapping[str, np.ndarray]) -> str:
	return array_trace_sha256(arrays, ORACLE_KEYS)


def _require_array(
	arrays: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...], dtype,
) -> np.ndarray:
	value = np.asarray(arrays[name])
	if value.shape != shape or value.dtype != np.dtype(dtype):
		raise ValueError(
			f'{name} must be {np.dtype(dtype)} {shape}, got '
			f'{value.dtype} {value.shape}.'
		)
	if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
		raise ValueError(f'{name} contains non-finite values.')
	return value


def branch_codes(action_dim: int) -> np.ndarray:
	values = [0]
	for axis in range(action_dim):
		values.extend((axis + 1, -(axis + 1)))
	return np.asarray(values, dtype=np.int32)


def validate_group_arrays(
	arrays: Mapping[str, np.ndarray], *, role_count: int, action_dim: int,
	state_dim: int, physics_dim: int, anchor_step: int,
	branch_magnitude: float,
) -> dict[str, Any]:
	"""Validate one indivisible root/sibling/background group."""
	if not isinstance(arrays, Mapping) or set(arrays) != GROUP_KEYS:
		actual = sorted(arrays) if isinstance(arrays, Mapping) else type(arrays)
		raise ValueError(
			f'Group arrays must be exactly {sorted(GROUP_KEYS)!r}, got {actual!r}.'
		)
	for value, label in (
		(role_count, 'role_count'), (action_dim, 'action_dim'),
		(state_dim, 'state_dim'), (physics_dim, 'physics_dim'),
	):
		if not isinstance(value, int) or isinstance(value, bool) or value < 1:
			raise ValueError(f'{label} must be a positive integer.')
	if not np.isfinite(branch_magnitude) or not 0.0 < branch_magnitude <= 1.0:
		raise ValueError('branch_magnitude must be finite and in (0,1].')
	branches = 1 + 2 * action_dim
	policy_shapes = {
		'rgb': (9, 64, 64),
		'object': (role_count, 1770),
		'object_mask': (role_count, 3, 64, 64),
		'role_exists': (role_count,),
	}
	policy_dtypes = {
		'rgb': np.uint8, 'object': np.float32,
		'object_mask': np.bool_, 'role_exists': np.float32,
	}
	for condition in CONDITIONS:
		for field in POLICY_FIELDS:
			history = _require_array(
				arrays, f'{condition}__history__policy_{field}',
				(HISTORY_FRAMES,) + policy_shapes[field], policy_dtypes[field],
			)
			future = _require_array(
				arrays, f'{condition}__future__policy_{field}',
				(branches, HORIZON) + policy_shapes[field], policy_dtypes[field],
			)
			if field == 'role_exists':
				if not np.array_equal(history, np.ones_like(history)):
					raise ValueError('ROF history role_exists must be all one.')
				if not np.array_equal(future, np.ones_like(future)):
					raise ValueError('ROF future role_exists must be all one.')

	history_action = _require_array(
		arrays, 'history__action',
		(HISTORY_FRAMES - 1, action_dim), np.float32,
	)
	branch_action = _require_array(
		arrays, 'branch__action', (branches, HORIZON, action_dim), np.float32,
	)
	if np.any(history_action < -1.0) or np.any(history_action > 1.0):
		raise ValueError('History action is outside the scaled [-1,1] action space.')
	if np.any(branch_action < -1.0) or np.any(branch_action > 1.0):
		raise ValueError('Branch action is outside the scaled [-1,1] action space.')
	expected_codes = branch_codes(action_dim)
	codes = _require_array(arrays, 'branch__code', (branches,), np.int32)
	is_zero = _require_array(arrays, 'branch__is_zero', (branches,), np.bool_)
	if not np.array_equal(codes, expected_codes):
		raise ValueError(f'Branch codes {codes.tolist()} != {expected_codes.tolist()}.')
	if not np.array_equal(is_zero, codes == 0):
		raise ValueError('branch__is_zero disagrees with branch__code.')
	interventions = branch_action[:, 0]
	expected = np.zeros((branches, action_dim), dtype=np.float32)
	row = 1
	for axis in range(action_dim):
		expected[row, axis] = np.float32(branch_magnitude)
		expected[row + 1, axis] = np.float32(-branch_magnitude)
		row += 2
	if not np.array_equal(interventions, expected):
		raise ValueError('Executed intervention actions are not the balanced basis design.')
	if HORIZON > 1 and not np.array_equal(
		branch_action[:, 1:],
		np.broadcast_to(branch_action[0:1, 1:], branch_action[:, 1:].shape),
	):
		raise ValueError('Open-loop continuation actions differ across siblings.')
	if np.linalg.matrix_rank(interventions[1:].astype(np.float64)) != action_dim:
		raise ValueError('Nonzero intervention actions are not full column rank.')

	official = _require_array(
		arrays, 'oracle__official_state',
		(branches, HORIZON + 1, state_dim), np.float64,
	)
	physics = _require_array(
		arrays, 'oracle__physics_state',
		(branches, HORIZON + 1, physics_dim), np.float64,
	)
	gt = _require_array(
		arrays, 'oracle__gt_role_mask',
		(branches, HORIZON + 1, role_count, 64, 64), np.bool_,
	)
	visible = _require_array(
		arrays, 'oracle__gt_visible',
		(branches, HORIZON + 1, role_count), np.bool_,
	)
	absolute_step = _require_array(
		arrays, 'oracle__absolute_step', (HORIZON + 1,), np.int64,
	)
	hard_official = _require_array(
		arrays, 'oracle__hard_official_state',
		(branches, HORIZON + 1, state_dim), np.float64,
	)
	hard_gt = _require_array(
		arrays, 'oracle__hard_gt_role_mask',
		(branches, HORIZON + 1, role_count, 64, 64), np.bool_,
	)
	clean_prefix_action = _require_array(
		arrays, 'oracle__clean_prefix_action',
		(branches, anchor_step, action_dim), np.float32,
	)
	hard_prefix_action = _require_array(
		arrays, 'oracle__hard_prefix_action',
		(branches, anchor_step, action_dim), np.float32,
	)
	full_shape = (branches, anchor_step + HORIZON + 1, physics_dim)
	clean_full_physics = _require_array(
		arrays, 'oracle__clean_full_physics_state', full_shape, np.float64,
	)
	hard_full_physics = _require_array(
		arrays, 'oracle__hard_full_physics_state', full_shape, np.float64,
	)
	if not np.array_equal(
		absolute_step, np.arange(anchor_step, anchor_step + HORIZON + 1),
	):
		raise ValueError('Oracle absolute steps do not begin at the anchor.')
	if not np.array_equal(official[:, 0], np.broadcast_to(official[0, 0], official[:, 0].shape)):
		raise ValueError('Sibling official states differ at the common anchor.')
	if not np.array_equal(physics[:, 0], np.broadcast_to(physics[0, 0], physics[:, 0].shape)):
		raise ValueError('Sibling physics states differ at the common anchor.')
	if not np.array_equal(gt[:, 0], np.broadcast_to(gt[0, 0], gt[:, 0].shape)):
		raise ValueError('Sibling GT masks differ at the common anchor.')
	if not np.array_equal(hard_official, official):
		raise ValueError('Clean/hard official-state evidence differs.')
	if not np.array_equal(hard_gt, gt):
		raise ValueError('Clean/hard GT-mask evidence differs.')
	if not np.array_equal(clean_prefix_action, hard_prefix_action):
		raise ValueError('Clean/hard prefix-action evidence differs.')
	if not np.array_equal(
		clean_prefix_action,
		np.broadcast_to(clean_prefix_action[0:1], clean_prefix_action.shape),
	):
		raise ValueError('Sibling prefix-action evidence is not exactly shared.')
	if not np.array_equal(
		clean_prefix_action[0, -(HISTORY_FRAMES - 1):], history_action,
	):
		raise ValueError('History actions are not the tail of the stored prefix.')
	if not np.array_equal(clean_full_physics, hard_full_physics):
		raise ValueError('Clean/hard complete physics evidence differs.')
	if not np.array_equal(
		clean_full_physics[:, anchor_step:anchor_step + HORIZON + 1], physics,
	):
		raise ValueError('Published oracle physics is not the clean full-trace slice.')
	for condition, full_physics in (
		('clean', clean_full_physics), ('hard', hard_full_physics),
	):
		prefix = full_physics[:, :anchor_step + 1]
		if not np.array_equal(
			prefix, np.broadcast_to(prefix[0:1], prefix.shape),
		):
			raise ValueError(
				f'{condition} sibling prefix physics is not exactly shared.'
			)
	if np.any(gt.sum(axis=2) > 1):
		raise ValueError('Oracle role masks overlap.')
	measured_visible = gt.reshape(branches, HORIZON + 1, role_count, -1).any(axis=-1)
	if not np.array_equal(visible, measured_visible):
		raise ValueError('oracle__gt_visible disagrees with oracle__gt_role_mask.')
	# A real branch protocol must alter physics.  Requiring every nonzero basis
	# sibling to differ from zero at the first post-action state fails closed on
	# clipped, ignored, or accidentally non-executed interventions.
	for index in range(1, branches):
		if np.array_equal(physics[index, 1], physics[0, 1]):
			raise ValueError(
				f'Branch {int(codes[index])} did not cause immediate physical divergence.'
			)
	return {
		'model_input_trace_sha256': model_input_trace_sha256(arrays),
		'oracle_trace_sha256': oracle_trace_sha256(arrays),
		'anchor_physics_sha256': typed_array_sha256(
			'anchor_physics', physics[0, 0],
		),
		'future_physics_sha256': typed_array_sha256(
			'future_physics', physics,
		),
		'prefix_physics_trace_sha256': typed_array_sha256(
			'prefix_physics', clean_full_physics[0, :anchor_step + 1],
		),
		'prefix_action_trace_sha256': typed_array_sha256(
			'prefix_action', clean_prefix_action[0],
		),
		'continuation_action_trace_sha256': typed_array_sha256(
			'continuation_action', branch_action[0, 1:],
		),
		'clean_full_physics_trace_sha256': [
			typed_array_sha256('full_physics', row)
			for row in clean_full_physics
		],
		'hard_full_physics_trace_sha256': [
			typed_array_sha256('full_physics', row)
			for row in hard_full_physics
		],
	}


def _safe_asset_path(root: Path, relative: str) -> Path:
	if not isinstance(relative, str) or not relative:
		raise ValueError('Group path must be a non-empty relative string.')
	path = (root / relative).resolve()
	try:
		path.relative_to(root.resolve())
	except ValueError as exc:
		raise ValueError(f'Group path escapes dataset root: {relative!r}.') from exc
	if path.is_symlink():
		raise ValueError(f'Symlinked group assets are forbidden: {path}.')
	return path


def _sha(value: Any) -> bool:
	return (
		isinstance(value, str) and len(value) == 64
		and all(char in '0123456789abcdef' for char in value)
	)


def validate_dataset(manifest_path: Path) -> dict[str, Any]:
	"""Revalidate the entire published dataset and its causal guards."""
	manifest_path = Path(manifest_path).resolve()
	if not manifest_path.is_file() or manifest_path.is_symlink():
		raise FileNotFoundError(manifest_path)
	payload = json.loads(manifest_path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict) or payload.get('format') != FORMAT:
		raise ValueError('Real-action branch manifest format mismatch.')
	if payload.get('controller_training_authorized') is not False:
		raise ValueError('A branch dataset can never authorize controller training.')
	if any((
		payload.get('history_frames') != HISTORY_FRAMES,
		payload.get('horizon') != HORIZON,
		payload.get('policy_input_fields') != [
			f'policy_{field}' for field in POLICY_FIELDS
		],
	)):
		raise ValueError('Compact evaluator-facing dataset schema mismatch.')
	input_contract = payload.get('model_input_contract')
	if not isinstance(input_contract, dict) or any((
		input_contract.get('keys') != list(MODEL_INPUT_KEYS),
		input_contract.get('schema') != 'robust_object_field_v0_real_action_branch_v1',
		input_contract.get('oracle_namespace_excluded') is not True,
		input_contract.get('condition_id_excluded') is not True,
		input_contract.get('only_executed_actions') is not True,
	)):
		raise ValueError('Model input isolation contract mismatch.')
	oracle_contract = payload.get('oracle_contract')
	if not isinstance(oracle_contract, dict) or any((
		oracle_contract.get('namespace') != ORACLE_NAMESPACE,
		oracle_contract.get('keys') != list(ORACLE_KEYS),
		oracle_contract.get('never_model_input') is not True,
		oracle_contract.get('same_state_query_guard') != 'exact_before_after',
	)):
		raise ValueError('Oracle isolation contract mismatch.')
	protocol = payload.get('branch_protocol')
	if not isinstance(protocol, dict) or any((
		protocol.get('history_frames') != HISTORY_FRAMES,
		protocol.get('horizon') != HORIZON,
		protocol.get('branch_action_index') != BRANCH_ACTION_INDEX,
		protocol.get('prefix') != 'fresh_reset_exact_open_loop_replay',
		protocol.get('interventions') != 'zero_plus_minus_each_scaled_action_basis',
		protocol.get('continuation') != 'identical_open_loop_actions_across_siblings',
		protocol.get('fake_shuffled_action_futures') is not False,
		protocol.get('background_pairing') != 'independent_fresh_reset_replay_exact_physics',
		protocol.get('root_grouping') != 'all_siblings_and_background_twins_one_split_unit',
	)):
		raise ValueError('Real-action branch protocol mismatch.')
	task = payload.get('task')
	roles = payload.get('role_names')
	action_dim = payload.get('action_dim')
	state_dim = payload.get('state_dim')
	physics_dim = payload.get('physics_dim')
	branch_magnitude = payload.get('branch_magnitude')
	if task not in TASKS:
		raise ValueError(f'Unsupported branch task {task!r}.')
	if not isinstance(roles, list) or not roles or len(set(roles)) != len(roles):
		raise ValueError('role_names are malformed.')
	for value, label in (
		(action_dim, 'action_dim'), (state_dim, 'state_dim'),
		(physics_dim, 'physics_dim'),
	):
		if not isinstance(value, int) or isinstance(value, bool) or value < 1:
			raise ValueError(f'{label} must be positive.')
	if not isinstance(branch_magnitude, (int, float)):
		raise ValueError('branch_magnitude is malformed.')
	source = payload.get('source')
	if not isinstance(source, dict):
		raise ValueError('Source identity is missing.')
	for field in ('runtime_config', 'checkpoint'):
		path = Path(str(source.get(field, ''))).resolve()
		if not path.is_file() or path.is_symlink():
			raise FileNotFoundError(path)
		if file_sha256(path) != source.get(f'{field}_sha256'):
			raise ValueError(f'Source {field} identity mismatch.')
	if source.get('backend') != 'robust_object_field':
		raise ValueError('Source backend is not robust_object_field.')

	groups = payload.get('groups')
	if not isinstance(groups, list) or len(groups) < 3:
		raise ValueError('At least three root groups are required.')
	by_id = {}
	root = manifest_path.parent
	for record in groups:
		if not isinstance(record, dict):
			raise ValueError('Group record must be an object.')
		root_id = record.get('root_id')
		if (
			not isinstance(root_id, int) or isinstance(root_id, bool)
			or root_id < 0 or root_id in by_id
		):
			raise ValueError(f'Malformed root id {root_id!r}.')
		anchor_step = record.get('anchor_step')
		if not isinstance(anchor_step, int) or anchor_step < HISTORY_FRAMES - 1:
			raise ValueError('anchor_step is invalid.')
		path = _safe_asset_path(root, record.get('relative_path'))
		if not path.is_file() or file_sha256(path) != record.get('sha256'):
			raise ValueError(f'Group asset identity mismatch: {path}.')
		with np.load(path, allow_pickle=False) as archive:
			arrays = {name: archive[name] for name in archive.files}
		hashes = validate_group_arrays(
			arrays, role_count=len(roles), action_dim=action_dim,
			state_dim=state_dim, physics_dim=physics_dim,
			anchor_step=anchor_step, branch_magnitude=float(branch_magnitude),
		)
		for key, expected in hashes.items():
			if record.get(key) != expected:
				raise ValueError(f'{key} mismatch for root {root_id}.')
		guards = record.get('guards')
		required_true = (
			'fresh_reset_replay_only', 'no_state_snapshot_restore',
			'all_sibling_prefix_physics_exact',
			'all_sibling_history_inputs_exact_within_condition',
			'clean_hard_actions_exact_per_branch',
			'clean_hard_physics_exact_per_branch',
			'clean_hard_official_state_exact_per_branch',
			'clean_hard_gt_exact_per_branch',
			'balanced_full_rank_real_actions',
			'immediate_physical_divergence',
			'one_root_one_split',
		)
		if not isinstance(guards, dict) or any(
			guards.get(name) is not True for name in required_true
		):
			raise ValueError(f'Root {root_id} causal guards did not all pass.')
		for key in (
			'prefix_action_trace_sha256', 'prefix_physics_trace_sha256',
			'continuation_action_trace_sha256',
		):
			if not _sha(record.get(key)):
				raise ValueError(f'Root {root_id} is missing {key}.')
		clean_physics = record.get('clean_full_physics_trace_sha256')
		hard_physics = record.get('hard_full_physics_trace_sha256')
		if (
			not isinstance(clean_physics, list)
			or len(clean_physics) != 1 + 2 * action_dim
			or any(not _sha(value) for value in clean_physics)
			or hard_physics != clean_physics
		):
			raise ValueError(
				f'Root {root_id} clean/hard full-physics hash evidence mismatch.'
			)
		by_id[root_id] = record
	if set(by_id) != set(range(len(groups))):
		raise ValueError('Root ids must be contiguous from zero.')
	splits = payload.get('root_splits')
	if not isinstance(splits, dict) or set(splits) != {'train', 'validation', 'test'}:
		raise ValueError('Root split table must define train/validation/test.')
	seen = set()
	for split in ('train', 'validation', 'test'):
		indices = splits[split]
		if not isinstance(indices, list) or not indices:
			raise ValueError(f'{split} root split is empty.')
		if any(index not in by_id for index in indices) or seen.intersection(indices):
			raise ValueError('Root split indices are invalid or overlap.')
		if any(by_id[index].get('split') != split for index in indices):
			raise ValueError('Group split disagrees with root split table.')
		seen.update(indices)
	if seen != set(by_id):
		raise ValueError('Root split table does not cover every group.')
	return payload


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.stem}.{uuid.uuid4().hex}.tmp.npz')
	try:
		with temporary.open('xb') as handle:
			np.savez_compressed(handle, **{
				key: np.ascontiguousarray(arrays[key]) for key in sorted(arrays)
			})
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()
	return file_sha256(path)


def _torch_observation_arrays(observation) -> dict[str, np.ndarray]:
	try:
		keys = set(observation.keys())
	except AttributeError as exc:
		raise RuntimeError('ROF observation must be a keyed TensorDict.') from exc
	if keys != {'rgb', 'object', 'object_mask', 'role_exists'}:
		raise RuntimeError(f'Unexpected ROF observation keys: {keys!r}.')
	result = {}
	for source, dtype in (
		('rgb', np.uint8), ('object', np.float32),
		('object_mask', np.bool_), ('role_exists', np.float32),
	):
		value = observation[source].detach().cpu().contiguous().numpy()
		result[source] = np.array(value, dtype=dtype, order='C', copy=True)
	return result


def _walk_children(env):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		yield current
		namespace = vars(current)
		current = namespace.get('env', namespace.get('_env'))


def _find_dm_control_source(env):
	for current in _walk_children(env):
		task = getattr(current, 'task', None)
		physics = getattr(current, 'physics', None)
		if task is not None and physics is not None and callable(
			getattr(task, 'get_observation', None)
		):
			return current
	raise RuntimeError('Cannot find official dm-control task/physics source.')


def _find_camera_id(env) -> int:
	for current in _walk_children(env):
		if 'camera_id' in vars(current):
			value = vars(current)['camera_id']
			if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
				return value
	raise RuntimeError('Cannot find concrete policy camera id.')


def _flatten_official_state(
	observation: Mapping[str, Any], schema: list[dict[str, Any]] | None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
	if not isinstance(observation, Mapping) or not observation:
		raise RuntimeError('Official dm-control state must be a non-empty mapping.')
	actual_schema, names, chunks = [], [], []
	offset = 0
	for field, raw in observation.items():
		value = np.asarray(raw, dtype=np.float64)
		if not np.isfinite(value).all():
			raise RuntimeError(f'Official state field {field!r} is non-finite.')
		flat = value.reshape(-1)
		actual_schema.append({
			'name': str(field), 'shape': list(value.shape), 'start': offset,
			'stop': offset + int(flat.size),
		})
		if value.ndim == 0:
			names.append(str(field))
		else:
			for index in np.ndindex(value.shape):
				names.append(f'{field}[{",".join(str(i) for i in index)}]')
		chunks.append(flat)
		offset += int(flat.size)
	if schema is not None and actual_schema != schema:
		raise RuntimeError('Official state schema changed across replays.')
	return (
		np.ascontiguousarray(np.concatenate(chunks), dtype=np.float64),
		actual_schema, names,
	)


def _gt_role_masks(physics, *, camera_id: int, selections) -> np.ndarray:
	try:
		from dm_control.mujoco.wrapper.mjbindings import enums
	except ImportError as exc:
		raise RuntimeError('MuJoCo segmentation constants are unavailable.') from exc
	segmentation = np.asarray(physics.render(
		height=64, width=64, camera_id=int(camera_id), segmentation=True,
	))
	if segmentation.shape != (64, 64, 2):
		raise RuntimeError(f'Unexpected MuJoCo segmentation shape {segmentation.shape}.')
	types = {
		'geom': int(enums.mjtObj.mjOBJ_GEOM),
		'site': int(enums.mjtObj.mjOBJ_SITE),
	}
	masks = []
	for selected in selections:
		mask = np.zeros((64, 64), dtype=np.bool_)
		for object_type, object_id in selected:
			mask |= (
				(segmentation[..., 0] == int(object_id))
				& (segmentation[..., 1] == types[object_type])
			)
		masks.append(mask)
	result = np.ascontiguousarray(np.stack(masks), dtype=np.bool_)
	if np.any(result.sum(axis=0) > 1):
		raise RuntimeError('Configured GT role masks overlap.')
	return result


def _capture_oracle(source, *, camera_id, selections, state_schema):
	physics = source.physics
	before = np.array(physics.get_state(), dtype=np.float64, copy=True)
	time_before = float(physics.time())
	official, schema, names = _flatten_official_state(
		source.task.get_observation(physics), state_schema,
	)
	masks = _gt_role_masks(physics, camera_id=camera_id, selections=selections)
	after = np.array(physics.get_state(), dtype=np.float64, copy=True)
	if not np.array_equal(before, after) or float(physics.time()) != time_before:
		raise RuntimeError('Oracle queries changed simulator physics.')
	return official, before, masks, schema, names


def _digest_sequence(name: str, rows: Sequence[np.ndarray]) -> str:
	return typed_array_sha256(name, np.stack(rows, axis=0))


def _make_cfg(args, raw, evaluator, *, condition, env_seed, background_seed):
	training_condition = 'hard' if raw.get('video_background_enabled') is True else 'clean'
	prepare = argparse.Namespace(
		task=args.task,
		backend='robust_object_field',
		erosion_pixels=0,
		training_seed=int(raw['seed']),
		expected_training_steps=int(raw['steps']),
		expected_training_eval_freq=int(raw['eval_freq']),
		expected_training_eval_episodes=int(raw['eval_episodes']),
		training_condition=training_condition,
		condition=condition,
		env_seed=int(env_seed),
		background_seed=int(background_seed),
		episodes=20,
		checkpoint=args.checkpoint,
		output=args.output_root / '.collector_runtime.json',
	)
	return evaluator._prepare(prepare, raw)


def _environment_context(env, *, expected_roles, task_spec, catalog_fn, select_fn):
	source = _find_dm_control_source(env)
	camera_id = _find_camera_id(env)
	catalog = catalog_fn(source.physics)
	selections = tuple(select_fn(catalog, selector) for selector in task_spec.selectors)
	identities = [identity for selected in selections for identity in selected]
	if len(identities) != len(set(identities)):
		raise RuntimeError('Configured privileged role selectors overlap.')
	if tuple(task_spec.roles) != tuple(expected_roles):
		raise RuntimeError(
			f'Task selector roles {task_spec.roles!r} != checkpoint roles '
			f'{tuple(expected_roles)!r}.'
		)
	return source, camera_id, selections


def _step_numpy(env, action: np.ndarray):
	import torch
	value = np.array(action, dtype=np.float32, order='C', copy=True)
	return env.step(torch.from_numpy(value))


def _generate_policy_prefix(
	args, raw, evaluator, make_env, set_seed, agent, *, env_seed,
	background_seed, planner_seed, anchor_step, expected_roles,
) -> tuple[np.ndarray, str]:
	"""Generate one policy-relevant prefix, then freeze it for all replays."""
	cfg = _make_cfg(
		args, raw, evaluator, condition='clean', env_seed=env_seed,
		background_seed=background_seed,
	)
	set_seed(env_seed)
	env = make_env(cfg)
	try:
		observation = env.reset()
		source = _find_dm_control_source(env)
		physics_rows = [np.array(source.physics.get_state(), dtype=np.float64, copy=True)]
		set_seed(planner_seed)
		agent._prev_mean.zero_()
		actions = []
		for step in range(anchor_step):
			try:
				import torch
				torch.compiler.cudagraph_mark_step_begin()
			except (AttributeError, RuntimeError):
				pass
			action = agent.act(observation, t0=step == 0, eval_mode=True)
			array = np.asarray(
				action.detach().cpu().numpy(), dtype=np.float32,
			).reshape(-1)
			actions.append(np.array(array, copy=True))
			observation, _, done, _ = env.step(action.detach().cpu())
			if done:
				raise RuntimeError(f'Canonical prefix terminated at step {step + 1}.')
			physics_rows.append(
				np.array(source.physics.get_state(), dtype=np.float64, copy=True)
			)
		action_array = np.ascontiguousarray(np.stack(actions), dtype=np.float32)
		return action_array, _digest_sequence('prefix_physics', physics_rows)
	finally:
		close = getattr(env, 'close', None)
		if callable(close):
			close()


def _build_branch_actions(
	*, action_dim: int, magnitude: float, continuation_rng,
) -> tuple[np.ndarray, np.ndarray]:
	codes = branch_codes(action_dim)
	branches = len(codes)
	result = np.empty((branches, HORIZON, action_dim), dtype=np.float32)
	result[:, 0] = 0.0
	row = 1
	for axis in range(action_dim):
		result[row, 0, axis] = np.float32(magnitude)
		result[row + 1, 0, axis] = np.float32(-magnitude)
		row += 2
	continuation = continuation_rng.uniform(
		-1.0, 1.0, size=(HORIZON - 1, action_dim),
	).astype(np.float32)
	result[:, 1:] = continuation[None]
	return result, codes


def _replay_one(
	args, raw, evaluator, make_env, set_seed, *, condition, env_seed,
	background_seed, prefix_actions, future_actions,
	anchor_step, expected_roles, task_spec, catalog_fn, select_fn,
	state_schema,
) -> dict[str, Any]:
	"""Fresh-reset replay of one real sibling/background trajectory."""
	cfg = _make_cfg(
		args, raw, evaluator, condition=condition, env_seed=env_seed,
		background_seed=background_seed,
	)
	# Use the exact canonical reset/global seed.  A distinct "replay seed" can
	# silently perturb wrappers that consult global RNG even when dm-control's
	# task RNG is configured correctly.
	set_seed(env_seed)
	env = make_env(cfg)
	try:
		observation = env.reset()
		source, camera_id, selections = _environment_context(
			env, expected_roles=expected_roles, task_spec=task_spec,
			catalog_fn=catalog_fn, select_fn=select_fn,
		)
		start_step = anchor_step - (HISTORY_FRAMES - 1)
		end_step = anchor_step + HORIZON
		policy_rows = {field: [] for field in POLICY_FIELDS}
		official_rows, physics_rows, gt_rows = [], [], []
		prefix_physics_rows = []
		full_physics_rows = []
		observed_schema = state_schema
		observed_names = None
		for absolute_step in range(end_step + 1):
			physics_now = np.array(
				source.physics.get_state(), dtype=np.float64, copy=True,
			)
			full_physics_rows.append(physics_now)
			if absolute_step <= anchor_step:
				prefix_physics_rows.append(physics_now)
			if absolute_step >= start_step:
				policy = _torch_observation_arrays(observation)
				for field in POLICY_FIELDS:
					policy_rows[field].append(policy[field])
				official, physics, masks, schema, names = _capture_oracle(
					source, camera_id=camera_id, selections=selections,
					state_schema=observed_schema,
				)
				if observed_schema is None:
					observed_schema = schema
					observed_names = names
				elif observed_names is None:
					observed_names = names
				elif names != observed_names:
					raise RuntimeError('Official state names changed during replay.')
				official_rows.append(official)
				physics_rows.append(physics)
				gt_rows.append(masks)
			if absolute_step == end_step:
				break
			action = (
				prefix_actions[absolute_step]
				if absolute_step < anchor_step
				else future_actions[absolute_step - anchor_step]
			)
			observation, _, done, _ = _step_numpy(env, action)
			if done:
				raise RuntimeError(
					f'{condition} replay terminated at absolute step {absolute_step + 1}.'
				)
		sequence = {
			field: np.ascontiguousarray(np.stack(policy_rows[field]))
			for field in POLICY_FIELDS
		}
		official = np.ascontiguousarray(np.stack(official_rows), dtype=np.float64)
		physics = np.ascontiguousarray(np.stack(physics_rows), dtype=np.float64)
		gt = np.ascontiguousarray(np.stack(gt_rows), dtype=np.bool_)
		return {
			'history': {
				field: sequence[field][:HISTORY_FRAMES] for field in POLICY_FIELDS
			},
			'future': {
				field: sequence[field][HISTORY_FRAMES:] for field in POLICY_FIELDS
			},
			'oracle_official': official[HISTORY_FRAMES - 1:],
			'oracle_physics': physics[HISTORY_FRAMES - 1:],
			'oracle_gt': gt[HISTORY_FRAMES - 1:],
			'state_schema': observed_schema,
			'state_names': observed_names,
			'camera_id': camera_id,
			'full_physics': np.ascontiguousarray(
				np.stack(full_physics_rows), dtype=np.float64,
			),
			'prefix_physics_trace_sha256': _digest_sequence(
				'prefix_physics', prefix_physics_rows,
			),
			'full_physics_trace_sha256': _digest_sequence(
				'full_physics', full_physics_rows,
			),
		}
	finally:
		close = getattr(env, 'close', None)
		if callable(close):
			close()


def _exact_nested(left: Any, right: Any) -> bool:
	if isinstance(left, dict) and isinstance(right, dict):
		return set(left) == set(right) and all(
			_exact_nested(left[key], right[key]) for key in left
		)
	return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)


def _validate_args(args) -> None:
	if args.task not in TASKS:
		raise ValueError(f'Unsupported real-branch task {args.task!r}.')
	for path in (args.runtime_config, args.checkpoint):
		if not path.is_file() or path.is_symlink():
			raise FileNotFoundError(path)
	expected_checkpoint = args.runtime_config.resolve().parent / 'models' / 'final.pt'
	if args.checkpoint.resolve() != expected_checkpoint:
		raise ValueError(
			f'Real branches require the final checkpoint paired with its runtime: '
			f'{args.checkpoint} != {expected_checkpoint}.'
		)
	stage = Path(str(args.output_root) + '.incomplete')
	if args.output_root.exists() or stage.exists():
		raise FileExistsError(args.output_root)
	if args.roots < 3 or args.train_roots < 1 or args.validation_roots < 1:
		raise ValueError('Require >=3 roots and nonempty train/validation splits.')
	if args.train_roots + args.validation_roots >= args.roots:
		raise ValueError('At least one held-out test root is required.')
	if not np.isfinite(args.branch_magnitude) or not 0.0 < args.branch_magnitude <= 1.0:
		raise ValueError('--branch-magnitude must be in (0,1].')
	anchors = tuple(int(item) for item in args.anchor_steps.split(','))
	if not anchors or any(
		step < HISTORY_FRAMES - 1 or step + HORIZON > 499 for step in anchors
	):
		raise ValueError('Anchor steps must support history and finish before step 500.')
	args.anchor_schedule = anchors
	seed_values = (
		args.env_seed_base, args.background_seed_base, args.planner_seed_base,
		args.continuation_seed_base,
	)
	if len(set(seed_values)) != len(seed_values):
		raise ValueError('Environment/background/planner/continuation seeds must differ.')


def collect(args) -> dict[str, Any]:
	"""Collect and atomically publish all strict root groups."""
	import torch
	from common.seed import set_seed
	from envs import make_env
	from isi_wm import TDMPC2
	from isi_wm.tools import evaluate_cutie_multitask_checkpoint as evaluator
	from isi_wm.tools.collect_cutie_multitask_support import (
		TASK_BY_NAME, _catalog, _selected_objects,
	)

	_validate_args(args)
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is required for the frozen ROF/Cutie checkpoint.')
	raw = evaluator._json(args.runtime_config)
	if raw.get('task') != args.task:
		raise ValueError(f'Runtime task {raw.get("task")!r} != {args.task!r}.')
	roles = tuple(raw.get('cutie_object_role_names', ()))
	if args.task not in TASK_BY_NAME or tuple(TASK_BY_NAME[args.task].roles) != roles:
		raise RuntimeError('Checkpoint roles do not match the immutable task selector.')
	task_spec = TASK_BY_NAME[args.task]
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')
	stage = Path(str(args.output_root) + '.incomplete')
	stage.mkdir(parents=True, exist_ok=False)

	# Construct one reference environment first so make_env writes the exact
	# observation/action dimensions into cfg before TDMPC2 is instantiated.
	reference_cfg = _make_cfg(
		args, raw, evaluator, condition='clean', env_seed=args.env_seed_base,
		background_seed=args.background_seed_base,
	)
	set_seed(args.env_seed_base)
	reference_env = make_env(reference_cfg)
	try:
		reference_env.reset()
		action_dim = int(reference_cfg.action_dim)
		if action_dim < 1:
			raise RuntimeError('Action dimension must be positive.')
		agent = TDMPC2(reference_cfg)
		agent.load(args.checkpoint)
		agent.eval()
	finally:
		reference_env.close()

	groups = []
	state_schema = None
	state_names = None
	physics_dim = None
	try:
		for root_id in range(args.roots):
			env_seed = args.env_seed_base + root_id
			background_seed = args.background_seed_base + root_id
			planner_seed = args.planner_seed_base + root_id
			continuation_seed = args.continuation_seed_base + root_id
			anchor_step = args.anchor_schedule[root_id % len(args.anchor_schedule)]
			prefix_actions, canonical_prefix_physics_sha = _generate_policy_prefix(
				args, raw, evaluator, make_env, set_seed, agent,
				env_seed=env_seed, background_seed=background_seed,
				planner_seed=planner_seed, anchor_step=anchor_step,
				expected_roles=roles,
			)
			if prefix_actions.shape != (anchor_step, action_dim):
				raise RuntimeError('Canonical prefix action shape changed.')
			branch_actions, codes = _build_branch_actions(
				action_dim=action_dim, magnitude=args.branch_magnitude,
				continuation_rng=np.random.default_rng(continuation_seed),
			)
			trajectories = {condition: [] for condition in CONDITIONS}
			for condition in CONDITIONS:
				for branch_index in range(len(codes)):
					trajectory = _replay_one(
						args, raw, evaluator, make_env, set_seed,
						condition=condition, env_seed=env_seed,
						background_seed=background_seed,
						prefix_actions=prefix_actions,
						future_actions=branch_actions[branch_index],
						anchor_step=anchor_step, expected_roles=roles,
						task_spec=task_spec, catalog_fn=_catalog,
						select_fn=_selected_objects, state_schema=state_schema,
					)
					if state_schema is None:
						state_schema = trajectory['state_schema']
						state_names = trajectory['state_names']
					elif (
						trajectory['state_schema'] != state_schema
						or trajectory['state_names'] != state_names
					):
						raise RuntimeError('Official state schema changed across roots.')
					observed_physics_dim = int(trajectory['oracle_physics'].shape[-1])
					if physics_dim is None:
						physics_dim = observed_physics_dim
					elif physics_dim != observed_physics_dim:
						raise RuntimeError('Physics state dimension changed across roots.')
					if trajectory['prefix_physics_trace_sha256'] != canonical_prefix_physics_sha:
						raise RuntimeError(
						f'{condition}/{int(codes[branch_index])} prefix physics '
						'differs from the canonical policy prefix.'
					)
					trajectories[condition].append(trajectory)

			# Histories are identical across real siblings within each visual
			# condition.  Across conditions only physics is required to match.
			for condition in CONDITIONS:
				reference = trajectories[condition][0]['history']
				if any(
					not _exact_nested(reference, row['history'])
					for row in trajectories[condition][1:]
				):
					raise RuntimeError(
						f'{condition} sibling ROF histories are not byte-identical.'
					)
			for branch_index, code in enumerate(codes):
				clean = trajectories['clean'][branch_index]
				hard = trajectories['hard'][branch_index]
				for key in ('oracle_physics', 'oracle_official', 'oracle_gt'):
					if not np.array_equal(clean[key], hard[key]):
						raise RuntimeError(
							f'Clean/hard {key} mismatch for real branch {int(code)}.'
						)
				if clean['full_physics_trace_sha256'] != hard['full_physics_trace_sha256']:
					raise RuntimeError('Clean/hard full physics traces differ.')

			arrays = {
				'history__action': np.ascontiguousarray(
					prefix_actions[anchor_step - (HISTORY_FRAMES - 1):anchor_step],
					dtype=np.float32,
				),
				'branch__action': branch_actions,
				'branch__code': np.ascontiguousarray(codes, dtype=np.int32),
				'branch__is_zero': np.ascontiguousarray(codes == 0, dtype=np.bool_),
				'oracle__official_state': np.ascontiguousarray(np.stack([
					row['oracle_official'] for row in trajectories['clean']
				]), dtype=np.float64),
				'oracle__physics_state': np.ascontiguousarray(np.stack([
					row['oracle_physics'] for row in trajectories['clean']
				]), dtype=np.float64),
				'oracle__gt_role_mask': np.ascontiguousarray(np.stack([
					row['oracle_gt'] for row in trajectories['clean']
				]), dtype=np.bool_),
				'oracle__hard_official_state': np.ascontiguousarray(np.stack([
					row['oracle_official'] for row in trajectories['hard']
				]), dtype=np.float64),
				'oracle__hard_gt_role_mask': np.ascontiguousarray(np.stack([
					row['oracle_gt'] for row in trajectories['hard']
				]), dtype=np.bool_),
				'oracle__clean_prefix_action': np.ascontiguousarray(
					np.broadcast_to(
						prefix_actions[None], (len(codes),) + prefix_actions.shape,
					).copy(), dtype=np.float32,
				),
				'oracle__hard_prefix_action': np.ascontiguousarray(
					np.broadcast_to(
						prefix_actions[None], (len(codes),) + prefix_actions.shape,
					).copy(), dtype=np.float32,
				),
				'oracle__clean_full_physics_state': np.ascontiguousarray(np.stack([
					row['full_physics'] for row in trajectories['clean']
				]), dtype=np.float64),
				'oracle__hard_full_physics_state': np.ascontiguousarray(np.stack([
					row['full_physics'] for row in trajectories['hard']
				]), dtype=np.float64),
				'oracle__absolute_step': np.arange(
					anchor_step, anchor_step + HORIZON + 1, dtype=np.int64,
				),
			}
			arrays['oracle__gt_visible'] = np.ascontiguousarray(
				arrays['oracle__gt_role_mask']
				.reshape(len(codes), HORIZON + 1, len(roles), -1).any(axis=-1),
				dtype=np.bool_,
			)
			for condition in CONDITIONS:
				for field in POLICY_FIELDS:
					arrays[f'{condition}__history__policy_{field}'] = np.ascontiguousarray(
						trajectories[condition][0]['history'][field]
					)
					arrays[f'{condition}__future__policy_{field}'] = np.ascontiguousarray(
						np.stack([
							row['future'][field] for row in trajectories[condition]
						], axis=0)
					)
			hashes = validate_group_arrays(
				arrays, role_count=len(roles), action_dim=action_dim,
				state_dim=len(state_names), physics_dim=int(physics_dim),
				anchor_step=anchor_step,
				branch_magnitude=float(args.branch_magnitude),
			)
			relative = Path('groups') / f'root_{root_id:04d}.npz'
			sha = _atomic_npz(stage / relative, arrays)
			split = (
				'train' if root_id < args.train_roots else
				'validation' if root_id < args.train_roots + args.validation_roots
				else 'test'
			)
			groups.append({
				'root_id': root_id,
				'split': split,
				'anchor_step': anchor_step,
				'env_seed': env_seed,
				'background_seed': background_seed,
				'planner_seed': planner_seed,
				'replay_reset_seed': env_seed,
				'continuation_seed': continuation_seed,
				'relative_path': relative.as_posix(),
				'sha256': sha,
				'prefix_action_trace_sha256': typed_array_sha256(
					'prefix_action', prefix_actions,
				),
				'prefix_physics_trace_sha256': canonical_prefix_physics_sha,
				'continuation_action_trace_sha256': typed_array_sha256(
					'continuation_action', branch_actions[0, 1:],
				),
				'clean_full_physics_trace_sha256': [
					row['full_physics_trace_sha256']
					for row in trajectories['clean']
				],
				'hard_full_physics_trace_sha256': [
					row['full_physics_trace_sha256']
					for row in trajectories['hard']
				],
				**hashes,
				'guards': {
					'fresh_reset_replay_only': True,
					'no_state_snapshot_restore': True,
					'all_sibling_prefix_physics_exact': True,
					'all_sibling_history_inputs_exact_within_condition': True,
					'clean_hard_actions_exact_per_branch': True,
					'clean_hard_physics_exact_per_branch': True,
					'clean_hard_official_state_exact_per_branch': True,
					'clean_hard_gt_exact_per_branch': True,
					'balanced_full_rank_real_actions': True,
					'immediate_physical_divergence': True,
					'one_root_one_split': True,
				},
			})
			print('ROF_REAL_ACTION_BRANCH_ROOT', json.dumps({
				'task': args.task, 'root_id': root_id, 'split': split,
				'anchor_step': anchor_step, 'branches': len(codes),
			}, allow_nan=False), flush=True)

		if state_schema is None or state_names is None or physics_dim is None:
			raise RuntimeError('No complete real-action root was collected.')
		root_splits = {
			split: [row['root_id'] for row in groups if row['split'] == split]
			for split in ('train', 'validation', 'test')
		}
		manifest = {
			'format': FORMAT,
			'status': 'rof_same_state_action_branch_dataset_complete',
			'controller_training_authorized': False,
			'task': args.task,
			'conditions': list(CONDITIONS),
			'history_frames': HISTORY_FRAMES,
			'horizon': HORIZON,
			'policy_input_fields': [f'policy_{field}' for field in POLICY_FIELDS],
			'role_names': list(roles),
			'role_count': len(roles),
			'action_dim': action_dim,
			'state_dim': len(state_names),
			'physics_dim': int(physics_dim),
			'branch_magnitude': float(args.branch_magnitude),
			'source': {
				'runtime_config': str(args.runtime_config.resolve()),
				'runtime_config_sha256': file_sha256(args.runtime_config),
				'checkpoint': str(args.checkpoint.resolve()),
				'checkpoint_sha256': file_sha256(args.checkpoint),
				'checkpoint_step': int(raw['steps']),
				'backend': 'robust_object_field',
				'policy_prefix_source': 'frozen_checkpoint_clean_eval_policy',
			},
			'model_input_contract': {
				'schema': 'robust_object_field_v0_real_action_branch_v1',
				'keys': list(MODEL_INPUT_KEYS),
				'oracle_namespace_excluded': True,
				'condition_id_excluded': True,
				'only_executed_actions': True,
			},
			'oracle_contract': {
				'namespace': ORACLE_NAMESPACE,
				'keys': list(ORACLE_KEYS),
				'never_model_input': True,
				'same_state_query_guard': 'exact_before_after',
				'official_state_source': 'dm_control.task.get_observation(current_physics)',
				'official_state_schema': state_schema,
				'official_state_names': state_names,
				'physics_state_source': 'dm_control.physics.get_state',
				'gt_role_mask_source': 'same_state_mujoco_segmentation_policy_camera',
			},
			'branch_protocol': {
				'history_frames': HISTORY_FRAMES,
				'horizon': HORIZON,
				'branch_action_index': BRANCH_ACTION_INDEX,
				'prefix': 'fresh_reset_exact_open_loop_replay',
				'interventions': 'zero_plus_minus_each_scaled_action_basis',
				'continuation': 'identical_open_loop_actions_across_siblings',
				'fake_shuffled_action_futures': False,
				'background_pairing': 'independent_fresh_reset_replay_exact_physics',
				'root_grouping': 'all_siblings_and_background_twins_one_split_unit',
			},
			'collection': {
				'roots': args.roots,
				'anchor_schedule': list(args.anchor_schedule),
				'camera_id': groups and trajectories['clean'][0]['camera_id'],
				'env_seed_base': args.env_seed_base,
				'background_seed_base': args.background_seed_base,
				'planner_seed_base': args.planner_seed_base,
				'continuation_seed_base': args.continuation_seed_base,
			},
			'root_splits': root_splits,
			'groups': groups,
		}
		manifest_path = stage / MANIFEST_NAME
		manifest_path.write_bytes(canonical_json_bytes(manifest))
		validate_dataset(manifest_path)
		os.replace(stage, args.output_root)
		return manifest
	except Exception:
		# Keep the incomplete directory for forensic diagnosis; never publish it.
		raise


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--output-root', type=Path, required=True)
	parser.add_argument('--roots', type=int, default=80)
	parser.add_argument('--train-roots', type=int, default=50)
	parser.add_argument('--validation-roots', type=int, default=10)
	parser.add_argument('--anchor-steps', default='80,160,240,320,400')
	parser.add_argument('--branch-magnitude', type=float, default=0.8)
	parser.add_argument('--env-seed-base', type=int, default=4242430)
	parser.add_argument('--background-seed-base', type=int, default=16180340)
	parser.add_argument('--planner-seed-base', type=int, default=86754000)
	parser.add_argument('--continuation-seed-base', type=int, default=27182810)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	manifest = collect(args)
	print('ROF_REAL_ACTION_BRANCH_DATASET_COMPLETE', json.dumps({
		'task': manifest['task'],
		'roots': len(manifest['groups']),
		'manifest': str((args.output_root / MANIFEST_NAME).resolve()),
	}, allow_nan=False), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

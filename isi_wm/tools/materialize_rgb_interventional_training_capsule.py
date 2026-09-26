"""Materialize strict RGB-only interventional training capsules.

The validated source dataset contains privileged simulator arrays because it
must prove that clean/hard observations are exact physical twins and that all
action siblings start from one common state.  Controller training must never
see those arrays.  This one-way materializer therefore:

1. re-runs the producing collector's complete validator;
2. selects whole root families from the source train split;
3. derives only an anonymous pairwise outcome-distance target from official
   state, using scale statistics fitted over those same train roots; and
4. publishes RGB observations, genuinely executed actions, branch codes, and
   the derived target in a new immutable directory.

The capsule contains neither object fields nor masks, roles, condition IDs,
raw simulator state, or any other oracle array.  It is a training-only side
input; policy evaluation remains RGB-only and does not load this capsule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for _path in (str(REPO_DIR), str(PROJECT_DIR)):
	while _path in sys.path:
		sys.path.remove(_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tools import collect_rof_real_action_branches as collector


FORMAT = 'rgb_interventional_training_capsule_v1'
STATUS = 'rgb_interventional_training_capsule_complete'
MANIFEST_NAME = 'training_capsule_manifest.json'
SOURCE_FORMAT = 'rof_same_state_action_branch_dataset_v1'
SOURCE_SPLIT = 'train'
SCREEN_TASKS = (
	'finger-spin', 'cartpole-swingup', 'reacher-easy',
	'cup-catch', 'walker-walk', 'acrobot-swingup',
)
RGB_SHAPE = (9, 64, 64)
SCALE_EPSILON = 1e-6
OUTCOME_GAP_THRESHOLD = 0.05
EXACT_REPLAY_NOISE_FLOOR = 0.0
SHARD_KEYS = frozenset((
	'clean__root_rgb', 'hard__root_rgb',
	'clean__future_rgb', 'hard__future_rgb',
	'branch__action', 'branch__code',
	'pair__outcome_gap', 'pair__eligible',
))
POLICY_INPUT_KEYS = (
	'clean__root_rgb', 'hard__root_rgb',
	'clean__future_rgb', 'hard__future_rgb',
	'branch__action', 'branch__code',
)
TRAINING_TARGET_KEYS = ('pair__outcome_gap', 'pair__eligible')
FORBIDDEN_KEY_FRAGMENTS = (
	'oracle', 'official_state', 'physics_state', 'mask', 'object', 'role',
	'proprio', 'joint', 'segmentation',
)


def _canonical_json_bytes(payload: Any) -> bytes:
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
	try:
		with temporary.open('xb') as handle:
			handle.write(payload)
			handle.flush()
			os.fsync(handle.fileno())
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _source_asset(manifest_path: Path, record: Mapping[str, Any]) -> Path:
	relative = record.get('relative_path')
	if not isinstance(relative, str) or not relative:
		raise ValueError('Source root record has no relative_path.')
	path = Path(relative)
	if path.is_absolute() or '..' in path.parts:
		raise ValueError('Source root asset path escapes its manifest root.')
	root = manifest_path.parent.resolve()
	resolved = (root / path).resolve()
	try:
		resolved.relative_to(root)
	except ValueError as exc:
		raise ValueError('Source root asset path escaped its manifest root.') from exc
	if not resolved.is_file() or resolved.is_symlink():
		raise FileNotFoundError(resolved)
	return resolved


def _capsule_asset(manifest_path: Path, relative: Any) -> Path:
	if not isinstance(relative, str) or not relative:
		raise ValueError('Capsule group path must be a non-empty relative string.')
	path = Path(relative)
	if path.is_absolute() or '..' in path.parts:
		raise ValueError('Capsule group path escapes its manifest root.')
	root = manifest_path.parent.resolve()
	resolved = (root / path).resolve()
	try:
		resolved.relative_to(root)
	except ValueError as exc:
		raise ValueError('Capsule group path escaped its manifest root.') from exc
	if resolved.is_symlink():
		raise ValueError(f'Symlinked capsule assets are forbidden: {resolved}.')
	return resolved


def _require_array(
	arrays: Mapping[str, np.ndarray], name: str,
	shape: tuple[int, ...], dtype,
) -> np.ndarray:
	if name not in arrays:
		raise ValueError(f'Capsule shard is missing {name!r}.')
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


class _OnlineCoordinateMoments:
	"""Numerically stable, vector-valued population moments."""

	def __init__(self):
		self.count = 0
		self.mean: np.ndarray | None = None
		self.m2: np.ndarray | None = None

	def update(self, rows: np.ndarray) -> None:
		value = np.asarray(rows, dtype=np.float64)
		if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
			raise ValueError('Official-state scale rows must be a non-empty matrix.')
		if not np.isfinite(value).all():
			raise ValueError('Official-state scale rows contain non-finite values.')
		batch_count = int(value.shape[0])
		batch_mean = value.mean(axis=0, dtype=np.float64)
		centered = value - batch_mean
		batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
		if self.mean is None:
			self.count = batch_count
			self.mean = batch_mean
			self.m2 = batch_m2
			return
		if value.shape[1] != self.mean.shape[0]:
			raise ValueError('Official-state dimension changed across train roots.')
		total = self.count + batch_count
		delta = batch_mean - self.mean
		self.mean = self.mean + delta * (batch_count / total)
		self.m2 = (
			self.m2 + batch_m2
			+ delta * delta * (self.count * batch_count / total)
		)
		self.count = total

	def scale(self, *, epsilon: float) -> np.ndarray:
		if self.mean is None or self.m2 is None or self.count < 1:
			raise ValueError('No train-root official-state outcomes were observed.')
		variance = np.maximum(self.m2 / self.count, 0.0)
		return np.ascontiguousarray(
			np.maximum(np.sqrt(variance), float(epsilon)), dtype=np.float64,
		)


def derive_pair_targets(
	official_state: np.ndarray, coordinate_scale: np.ndarray, *,
	threshold: float = OUTCOME_GAP_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return anonymous branch-pair outcome gaps and eligibility.

	``official_state`` includes the common anchor at time zero.  Only genuinely
	executed post-action states (1..H) contribute to the target.
	"""
	official = np.asarray(official_state, dtype=np.float64)
	scale = np.asarray(coordinate_scale, dtype=np.float64)
	if official.ndim != 3 or official.shape[0] < 3 or official.shape[1] < 2:
		raise ValueError('Official-state sibling tensor is malformed.')
	if scale.shape != (official.shape[2],):
		raise ValueError('Official-state coordinate scale has the wrong width.')
	if not np.isfinite(official).all() or not np.isfinite(scale).all():
		raise ValueError('Outcome target source contains non-finite values.')
	if np.any(scale < SCALE_EPSILON):
		raise ValueError('Outcome coordinate scale was not epsilon guarded.')
	if not np.isfinite(threshold) or threshold < EXACT_REPLAY_NOISE_FLOOR:
		raise ValueError('Outcome eligibility threshold is invalid.')
	# [branch, branch, horizon, coordinate]
	difference = (
		official[:, None, 1:, :] - official[None, :, 1:, :]
	) / scale[None, None, None, :]
	gap64 = np.sqrt(np.mean(difference * difference, axis=-1, dtype=np.float64))
	# Make the algebraic invariants exact after float32 conversion.
	gap64 = 0.5 * (gap64 + np.swapaxes(gap64, 0, 1))
	for index in range(gap64.shape[0]):
		gap64[index, index, :] = 0.0
	gap = np.ascontiguousarray(gap64, dtype=np.float32)
	off_diagonal = ~np.eye(gap.shape[0], dtype=np.bool_)[:, :, None]
	eligible = np.ascontiguousarray(
		off_diagonal & (gap > np.float32(threshold)), dtype=np.bool_,
	)
	return gap, eligible


def validate_shard(
	arrays: Mapping[str, np.ndarray], *, action_dim: int, horizon: int,
	branch_magnitude: float, outcome_gap_threshold: float,
) -> dict[str, int]:
	if not isinstance(arrays, Mapping) or set(arrays) != SHARD_KEYS:
		actual = sorted(arrays) if isinstance(arrays, Mapping) else type(arrays)
		raise ValueError(
			f'RGB capsule arrays must be exactly {sorted(SHARD_KEYS)!r}, '
			f'got {actual!r}.'
		)
	for key in arrays:
		lower = str(key).lower()
		if any(fragment in lower for fragment in FORBIDDEN_KEY_FRAGMENTS):
			raise ValueError(f'Forbidden privileged/object key entered capsule: {key!r}.')
	if (
		not isinstance(action_dim, int) or isinstance(action_dim, bool)
		or action_dim < 1
	):
		raise ValueError('action_dim must be a positive integer.')
	if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
		raise ValueError('horizon must be a positive integer.')
	if (
		not np.isfinite(branch_magnitude)
		or not 0.0 < float(branch_magnitude) <= 1.0
	):
		raise ValueError('branch_magnitude must lie in (0,1].')
	branches = 1 + 2 * action_dim
	for condition in ('clean', 'hard'):
		_require_array(arrays, f'{condition}__root_rgb', RGB_SHAPE, np.uint8)
		_require_array(
			arrays, f'{condition}__future_rgb',
			(branches, horizon) + RGB_SHAPE, np.uint8,
		)
	actions = _require_array(
		arrays, 'branch__action', (branches, horizon, action_dim), np.float32,
	)
	codes = _require_array(arrays, 'branch__code', (branches,), np.int32)
	expected_codes = branch_codes(action_dim)
	if not np.array_equal(codes, expected_codes):
		raise ValueError('Branch codes are not zero plus/minus every action basis.')
	expected_actions = np.zeros((branches, action_dim), dtype=np.float32)
	row = 1
	for axis in range(action_dim):
		expected_actions[row, axis] = np.float32(branch_magnitude)
		expected_actions[row + 1, axis] = np.float32(-branch_magnitude)
		row += 2
	if not np.array_equal(actions[:, 0], expected_actions):
		raise ValueError('First branch actions are not the declared real basis design.')
	if horizon > 1 and not np.array_equal(
		actions[:, 1:],
		np.broadcast_to(actions[0:1, 1:], actions[:, 1:].shape),
	):
		raise ValueError('Open-loop continuation actions differ across siblings.')
	if np.any(actions < -1.0) or np.any(actions > 1.0):
		raise ValueError('Branch action is outside the scaled [-1,1] range.')
	gap = _require_array(
		arrays, 'pair__outcome_gap', (branches, branches, horizon), np.float32,
	)
	eligible = _require_array(
		arrays, 'pair__eligible', (branches, branches, horizon), np.bool_,
	)
	if np.any(gap < 0.0) or not np.array_equal(gap, np.swapaxes(gap, 0, 1)):
		raise ValueError('Pair outcome gaps must be nonnegative and exactly symmetric.')
	diagonal = gap[np.arange(branches), np.arange(branches)]
	if not np.array_equal(diagonal, np.zeros_like(diagonal)):
		raise ValueError('Pair outcome gap diagonal must be exactly zero.')
	off_diagonal = ~np.eye(branches, dtype=np.bool_)[:, :, None]
	expected_eligible = off_diagonal & (
		gap > np.float32(outcome_gap_threshold)
	)
	if not np.array_equal(eligible, expected_eligible):
		raise ValueError('pair__eligible disagrees with the declared threshold.')
	if not np.array_equal(eligible, np.swapaxes(eligible, 0, 1)):
		raise ValueError('Pair eligibility must be exactly symmetric.')
	return {'branches': branches, 'horizon': horizon}


def _is_sha256(value: Any) -> bool:
	return (
		isinstance(value, str) and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def validate_capsule(manifest_path: Path) -> dict[str, Any]:
	"""Validate a published RGB capsule without reopening source privilege."""
	manifest_path = Path(manifest_path).resolve()
	if not manifest_path.is_file() or manifest_path.is_symlink():
		raise FileNotFoundError(manifest_path)
	payload = json.loads(manifest_path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict) or payload.get('format') != FORMAT:
		raise ValueError('RGB interventional capsule format mismatch.')
	if payload.get('status') != STATUS:
		raise ValueError('RGB interventional capsule is incomplete.')
	required_exact = {
		'controller_auxiliary_training_authorized': True,
		'source_split': SOURCE_SPLIT,
		'test_time_input': 'rgb_only',
		'condition_pair': ['clean', 'hard'],
		'exact_physical_twins': True,
		'model_input_only': True,
		'no_cutie': True,
		'cutie_arrays_present': False,
		'masks_present': False,
		'oracle_arrays_present': False,
		'raw_privileged_arrays_present': False,
		'object_arrays_present': False,
		'whole_root_families_only': True,
		'fake_shuffled_action_futures': False,
		'policy_input_keys': list(POLICY_INPUT_KEYS),
		'training_target_keys': list(TRAINING_TARGET_KEYS),
	}
	for key, expected in required_exact.items():
		if payload.get(key) != expected:
			raise ValueError(f'RGB capsule contract mismatch for {key}.')
	outcome_gap = payload.get('outcome_gap')
	if not isinstance(outcome_gap, dict) or outcome_gap != {
		'threshold': OUTCOME_GAP_THRESHOLD,
		'normalization': 'train_root_coordinate_population_std_then_rms',
		'exact_replay_noise_floor': EXACT_REPLAY_NOISE_FLOOR,
		'ineligible_margin': 0.0,
	}:
		raise ValueError('RGB capsule outcome-gap training contract is malformed.')
	source = payload.get('source_dataset')
	if not isinstance(source, dict) or any((
		source.get('format') != SOURCE_FORMAT,
		not _is_sha256(source.get('manifest_sha256')),
		source.get('controller_training_authorized') is not False,
		source.get('validation_performed_before_privilege_stripping') is not True,
		source.get('validation_performed_before_derivation_and_stripping') is not True,
	)):
		raise ValueError('RGB capsule source provenance is malformed.')
	background = payload.get('background_twin_contract')
	if not isinstance(background, dict) or any((
		background.get('conditions') != ['clean', 'hard'],
		background.get('exact_physics_per_branch') is not True,
		background.get('exact_official_state_per_branch') is not True,
		background.get('independent_fresh_reset_replay') is not True,
	)):
		raise ValueError('RGB capsule clean/hard twin contract is malformed.')
	action = payload.get('action_sibling_contract')
	if not isinstance(action, dict) or any((
		action.get('common_exact_physical_root') is not True,
		action.get('only_real_executed_actions') is not True,
		action.get('interventions') != 'zero_plus_minus_each_scaled_action_basis',
		action.get('continuation') != 'identical_open_loop_actions_across_siblings',
	)):
		raise ValueError('RGB capsule action-sibling contract is malformed.')
	target = payload.get('outcome_target_contract')
	if not isinstance(target, dict) or any((
		target.get('source') != 'oracle_official_state_train_roots_only',
		target.get('raw_source_stripped_after_derivation') is not True,
		target.get('coordinate_scale') != 'population_std_all_train_branch_futures',
		target.get('distance') != 'coordinate_standardized_rms',
		target.get('symmetric') is not True,
		target.get('diagonal_zero') is not True,
		target.get('scale_epsilon') != SCALE_EPSILON,
		target.get('eligibility_threshold') != OUTCOME_GAP_THRESHOLD,
		target.get('exact_replay_noise_floor') != EXACT_REPLAY_NOISE_FLOOR,
		not _is_sha256(target.get('coordinate_scale_sha256')),
		not isinstance(target.get('coordinate_count'), int),
		target.get('coordinate_count', 0) < 1,
		not isinstance(target.get('state_dim'), int),
		target.get('state_dim', 0) < 1,
	)):
		raise ValueError('RGB capsule outcome-target contract is malformed.')
	action_dim = payload.get('action_dim')
	horizon = payload.get('horizon')
	branch_magnitude = payload.get('branch_magnitude')
	if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
		raise ValueError('RGB capsule action_dim is malformed.')
	if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
		raise ValueError('RGB capsule horizon is malformed.')
	if not isinstance(branch_magnitude, (int, float)):
		raise ValueError('RGB capsule branch_magnitude is malformed.')
	groups = payload.get('groups')
	root_ids = payload.get('source_root_ids')
	if not isinstance(groups, list) or not groups:
		raise ValueError('RGB capsule has no train root families.')
	if not isinstance(root_ids, list) or len(root_ids) != len(groups):
		raise ValueError('RGB capsule source_root_ids are malformed.')
	seen = set()
	for record in groups:
		if not isinstance(record, dict):
			raise ValueError('RGB capsule group record is malformed.')
		root_id = record.get('root_id')
		if (
			not isinstance(root_id, int) or isinstance(root_id, bool)
			or root_id < 0 or root_id in seen
		):
			raise ValueError(f'Invalid or duplicate RGB capsule root {root_id!r}.')
		seen.add(root_id)
		if not _is_sha256(record.get('source_group_sha256')):
			raise ValueError(f'RGB capsule root {root_id} lost source binding.')
		asset = _capsule_asset(manifest_path, record.get('relative_path'))
		if not asset.is_file() or file_sha256(asset) != record.get('sha256'):
			raise ValueError(f'RGB capsule root {root_id} asset identity mismatch.')
		with np.load(asset, allow_pickle=False) as archive:
			arrays = {key: archive[key] for key in archive.files}
		validate_shard(
			arrays, action_dim=action_dim, horizon=horizon,
			branch_magnitude=float(branch_magnitude),
			outcome_gap_threshold=float(target['eligibility_threshold']),
		)
	if root_ids != [record['root_id'] for record in groups]:
		raise ValueError('RGB capsule root order disagrees with source_root_ids.')
	if set(root_ids) != seen:
		raise ValueError('RGB capsule did not bind each selected train root exactly once.')
	payload['_manifest_sha256'] = file_sha256(manifest_path)
	return payload


def _selected_train_records(source: Mapping[str, Any]) -> list[dict[str, Any]]:
	groups = source.get('groups')
	if not isinstance(groups, list):
		raise ValueError('Source branch dataset group table is malformed.')
	selected = [record for record in groups if record.get('split') == SOURCE_SPLIT]
	if not selected:
		raise ValueError('Source branch dataset has no train root families.')
	declared = source.get('root_splits', {}).get(SOURCE_SPLIT)
	root_ids = [record.get('root_id') for record in selected]
	if declared != root_ids:
		raise ValueError('Selected train roots disagree with source root_splits order.')
	if len(root_ids) != len(set(root_ids)):
		raise ValueError('Source train root ids are duplicated.')
	return selected


def _fit_coordinate_scale(
	source_manifest: Path, selected: list[dict[str, Any]], *,
	state_dim: int, branches: int, horizon: int,
) -> tuple[np.ndarray, int]:
	moments = _OnlineCoordinateMoments()
	for record in selected:
		asset = _source_asset(source_manifest, record)
		if file_sha256(asset) != record.get('sha256'):
			raise ValueError(
				f'Source root {record.get("root_id")} changed after validation.'
			)
		with np.load(asset, allow_pickle=False) as archive:
			official = np.asarray(archive['oracle__official_state'])
			if official.shape != (branches, horizon + 1, state_dim):
				raise ValueError(
					f'Source root {record.get("root_id")} official-state shape changed.'
				)
			if official.dtype != np.float64 or not np.isfinite(official).all():
				raise ValueError('Source official-state tensor is malformed.')
			moments.update(official[:, 1:, :].reshape(-1, state_dim))
	return moments.scale(epsilon=SCALE_EPSILON), moments.count


def materialize(args) -> dict[str, Any]:
	source_manifest = Path(args.source_manifest).resolve()
	output_root = Path(args.output_root).resolve()
	stage = output_root.with_name(output_root.name + '.incomplete')
	if output_root.exists() or stage.exists():
		raise FileExistsError(
			f'Refusing to overwrite RGB capsule output or stage: {output_root}'
		)
	if args.source_split != SOURCE_SPLIT:
		raise ValueError('Only complete source train roots may enter RGB capsules.')
	if args.authorize_auxiliary_training != 'I_UNDERSTAND_RGB_ONLY_WITH_DERIVED_TARGETS':
		raise ValueError(
			'Explicit --authorize-auxiliary-training '
			'I_UNDERSTAND_RGB_ONLY_WITH_DERIVED_TARGETS is required.'
		)

	# Revalidate every source shard and all exact-twin/action-sibling guards
	# before opening any privileged array.  Expand only the task allow-list in
	# this process; the producing validator retains all other strict contracts.
	collector.TASKS = SCREEN_TASKS
	collector.validate_dataset(source_manifest)
	source = json.loads(source_manifest.read_text(encoding='utf-8'))
	if source.get('format') != SOURCE_FORMAT:
		raise ValueError('Source branch dataset format changed after validation.')
	selected = _selected_train_records(source)
	action_dim = int(source['action_dim'])
	horizon = int(source['horizon'])
	state_dim = int(source['state_dim'])
	branches = 1 + 2 * action_dim
	coordinate_scale, coordinate_count = _fit_coordinate_scale(
		source_manifest, selected, state_dim=state_dim,
		branches=branches, horizon=horizon,
	)

	stage.mkdir(parents=True)
	groups = []
	try:
		for record in selected:
			asset = _source_asset(source_manifest, record)
			if file_sha256(asset) != record.get('sha256'):
				raise ValueError(
					f'Source root {record.get("root_id")} changed after scale fitting.'
				)
			with np.load(asset, allow_pickle=False) as archive:
				# This explicit allow-list is the privilege boundary.  Never copy by
				# archive iteration, prefix, or source namespace.
				arrays = {
					'clean__root_rgb': np.ascontiguousarray(
						archive['clean__history__policy_rgb'][-1]
					),
					'hard__root_rgb': np.ascontiguousarray(
						archive['hard__history__policy_rgb'][-1]
					),
					'clean__future_rgb': np.ascontiguousarray(
						archive['clean__future__policy_rgb']
					),
					'hard__future_rgb': np.ascontiguousarray(
						archive['hard__future__policy_rgb']
					),
					'branch__action': np.ascontiguousarray(archive['branch__action']),
					'branch__code': np.ascontiguousarray(archive['branch__code']),
				}
				gap, eligible = derive_pair_targets(
					archive['oracle__official_state'], coordinate_scale,
					threshold=OUTCOME_GAP_THRESHOLD,
				)
				arrays['pair__outcome_gap'] = gap
				arrays['pair__eligible'] = eligible
			validate_shard(
				arrays, action_dim=action_dim, horizon=horizon,
				branch_magnitude=float(source['branch_magnitude']),
				outcome_gap_threshold=OUTCOME_GAP_THRESHOLD,
			)
			relative = Path('groups') / f'root_{int(record["root_id"]):04d}.npz'
			groups.append({
				'root_id': int(record['root_id']),
				'relative_path': relative.as_posix(),
				'sha256': _atomic_npz(stage / relative, arrays),
				'source_group_sha256': str(record['sha256']),
			})

		manifest = {
			'format': FORMAT,
			'status': STATUS,
			'controller_auxiliary_training_authorized': True,
			'task': source['task'],
			'source_split': SOURCE_SPLIT,
			'test_time_input': 'rgb_only',
			'condition_pair': ['clean', 'hard'],
			'exact_physical_twins': True,
			# Here "model_input_only" excludes raw privileged measurements;
			# pair__* values are separately declared training targets and are
			# never policy inputs.
			'model_input_only': True,
			'no_cutie': True,
			'cutie_arrays_present': False,
			'masks_present': False,
			'oracle_arrays_present': False,
			'raw_privileged_arrays_present': False,
			'object_arrays_present': False,
			'whole_root_families_only': True,
			'fake_shuffled_action_futures': False,
			'outcome_gap': {
				'threshold': OUTCOME_GAP_THRESHOLD,
				'normalization': 'train_root_coordinate_population_std_then_rms',
				'exact_replay_noise_floor': EXACT_REPLAY_NOISE_FLOOR,
				'ineligible_margin': 0.0,
			},
			'policy_input_keys': list(POLICY_INPUT_KEYS),
			'training_target_keys': list(TRAINING_TARGET_KEYS),
			'action_dim': action_dim,
			'horizon': horizon,
			'branch_magnitude': float(source['branch_magnitude']),
			'source_root_ids': [int(record['root_id']) for record in selected],
			'source_dataset': {
				'format': source['format'],
				'manifest_sha256': file_sha256(source_manifest),
				'controller_training_authorized': source.get(
					'controller_training_authorized'
				),
				'validation_performed_before_privilege_stripping': True,
				'validation_performed_before_derivation_and_stripping': True,
			},
			'background_twin_contract': {
				'conditions': ['clean', 'hard'],
				'exact_physics_per_branch': True,
				'exact_official_state_per_branch': True,
				'independent_fresh_reset_replay': True,
			},
			'action_sibling_contract': {
				'common_exact_physical_root': True,
				'only_real_executed_actions': True,
				'interventions': 'zero_plus_minus_each_scaled_action_basis',
				'continuation': 'identical_open_loop_actions_across_siblings',
			},
			'outcome_target_contract': {
				'source': 'oracle_official_state_train_roots_only',
				'raw_source_stripped_after_derivation': True,
				'coordinate_scale': 'population_std_all_train_branch_futures',
				'distance': 'coordinate_standardized_rms',
				'symmetric': True,
				'diagonal_zero': True,
				'scale_epsilon': SCALE_EPSILON,
				'coordinate_count': int(coordinate_count),
				'state_dim': state_dim,
				'coordinate_scale_sha256': typed_array_sha256(
					'official_state_coordinate_scale', coordinate_scale,
				),
				'exact_replay_noise_floor': EXACT_REPLAY_NOISE_FLOOR,
				'eligibility_threshold': OUTCOME_GAP_THRESHOLD,
				'eligibility': 'off_diagonal_and_gap_strictly_greater_than_threshold',
			},
			'groups': groups,
		}
		manifest_path = stage / MANIFEST_NAME
		_atomic_bytes(manifest_path, _canonical_json_bytes(manifest))
		validate_capsule(manifest_path)
		os.replace(stage, output_root)
		return manifest
	except Exception:
		# Preserve the incomplete directory for forensic inspection; never
		# publish an output root whose validation did not complete.
		raise


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--source-manifest', type=Path, required=True)
	parser.add_argument('--output-root', type=Path, required=True)
	parser.add_argument('--source-split', default=SOURCE_SPLIT, choices=(SOURCE_SPLIT,))
	parser.add_argument('--authorize-auxiliary-training', required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	manifest = materialize(args)
	path = Path(args.output_root).resolve() / MANIFEST_NAME
	print('RGB_INTERVENTIONAL_TRAINING_CAPSULE_COMPLETE', json.dumps({
		'task': manifest['task'],
		'roots': len(manifest['groups']),
		'manifest': str(path),
		'manifest_sha256': file_sha256(path),
	}, allow_nan=False), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

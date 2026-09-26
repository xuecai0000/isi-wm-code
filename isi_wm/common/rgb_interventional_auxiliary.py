"""Training-only interventions for the official raw-RGB TD-MPC2 model.

The auxiliary capsule contains only deployable RGB histories, actions that were
actually executed from an exactly cloned simulator state, and a *derived*
outcome-gap/eligibility label.  Raw simulator state is consumed by the
materializer and is forbidden from this capsule.  Cutie outputs, masks, object
roles, rewards, and every oracle array are forbidden as well.

Two independent interventions are supported:

* clean/hard renders of the same physical state train background invariance;
* real action siblings train the official RGB encoder and the same dynamics
  used by TD-MPC2 planning.  Separation/ranking is enabled only when the
  privileged training-time outcome check says the two executed futures differ.

Nothing in this module changes the online observation or policy interface.
Disabled defaults leave legacy training, RNG, optimizer, and checkpoints alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from . import rgb_interventional_pairing


FORMAT = 'rgb_interventional_training_capsule_v1'
STATUS = 'rgb_interventional_training_capsule_complete'
CONDITIONS = ('clean', 'hard')
MAX_HORIZON = 5
SHARD_KEYS = frozenset({
	'clean__root_rgb', 'hard__root_rgb',
	'clean__future_rgb', 'hard__future_rgb',
	'branch__action', 'branch__code',
	'pair__outcome_gap', 'pair__eligible',
})
POLICY_INPUT_KEYS = (
	'clean__root_rgb', 'hard__root_rgb',
	'clean__future_rgb', 'hard__future_rgb',
	'branch__action', 'branch__code',
)
TRAINING_TARGET_KEYS = ('pair__outcome_gap', 'pair__eligible')
SHARED_FORWARD_CONTRACT = (
	'root_clean_hard_B_then_future_clean_hard_positive_negative_B_'
	'shared_pair_crop_then_positive_action_rollout_clean_hard_B_v1'
)

_IDENTITY_KEYS = (
	'rgb_interventional_aux_manifest',
	'rgb_interventional_aux_manifest_sha256',
	'rgb_interventional_aux_capsule_format',
	'rgb_interventional_aux_source_dataset_sha256',
)


def _get(cfg, key: str, default=None):
	return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def _strict_positive_int(value, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value < 1:
		raise ValueError(f'{name} must be a positive integer.')
	return int(value)


def _finite_nonnegative(value, name: str) -> float:
	if isinstance(value, bool):
		raise ValueError(f'{name} must be a finite non-negative number.')
	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f'{name} must be a finite non-negative number.') from exc
	if not math.isfinite(result) or result < 0.0:
		raise ValueError(f'{name} must be a finite non-negative number.')
	return result


def enabled(cfg) -> bool:
	value = _get(cfg, 'rgb_interventional_aux_enabled', False)
	if type(value) is not bool:
		raise ValueError('rgb_interventional_aux_enabled must be a strict boolean.')
	return value


def file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with Path(path).open('rb') as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def branch_codes(action_dim: int) -> np.ndarray:
	_strict_positive_int(action_dim, 'action_dim')
	values = [0]
	for axis in range(action_dim):
		values.extend((axis + 1, -(axis + 1)))
	return np.asarray(values, dtype=np.int32)


def balanced_ordered_code_pairs(action_dim: int) -> np.ndarray:
	"""Axis-balanced zero/basis/opposite-basis relations in both directions."""
	_strict_positive_int(action_dim, 'action_dim')
	pairs = []
	for axis in range(action_dim):
		positive, negative = axis + 1, -(axis + 1)
		for left, right in ((0, positive), (0, negative), (positive, negative)):
			pairs.extend(((left, right), (right, left)))
	return np.asarray(pairs, dtype=np.int64)


def validate_config(cfg, *, require_bound_manifest: bool = False) -> dict | None:
	"""Validate the four opt-in scientific arms and raw-RGB deployment contract."""
	if not enabled(cfg):
		rgb_interventional_pairing.config(
			cfg, base_arm='disabled', enabled=False,
		)
		for key in _IDENTITY_KEYS:
			if _get(cfg, key, None) is not None:
				raise ValueError(f'{key} requires rgb_interventional_aux_enabled=true.')
		return None

	if _get(cfg, 'obs', None) != 'rgb' or set(_get(cfg, 'obs_shape', {'rgb': ()})) != {'rgb'}:
		raise ValueError('RGB interventions require the official raw-RGB observation only.')
	if bool(_get(cfg, 'flat_anchor', False)):
		raise ValueError('RGB interventions forbid FlatAnchor and all Cutie branches.')
	if bool(_get(cfg, 'cutie_masked_rgb_enabled', False)):
		raise ValueError('RGB interventions forbid Cutie-masked RGB.')
	if bool(_get(cfg, 'cutie_mask_guided_rgb_enabled', False)):
		raise ValueError('RGB interventions forbid mask-guided RGB.')
	if bool(_get(cfg, 'robust_object_field_enabled', False)):
		raise ValueError('RGB interventions forbid ROF-WM.')
	for key in (
		'object_state_supervision_enabled',
		'object_state_supervision_collect_labels',
		'object_state_bottleneck_enabled',
	):
		if bool(_get(cfg, key, False)):
			raise ValueError(
			f'RGB interventions forbid simulator-state route {key}.'
		)
	if bool(_get(cfg, 'multitask', False)):
		raise ValueError('The first RGB interventional screen is single-task.')
	if bool(_get(cfg, 'compile', False)):
		raise ValueError('The paired RGB screen requires compile=false.')

	coefs = {
		'background': _finite_nonnegative(
			_get(cfg, 'rgb_interventional_aux_background_coef', 0.0),
			'rgb_interventional_aux_background_coef',
		),
		'positive': _finite_nonnegative(
			_get(cfg, 'rgb_interventional_aux_positive_coef', 0.0),
			'rgb_interventional_aux_positive_coef',
		),
		'separation': _finite_nonnegative(
			_get(cfg, 'rgb_interventional_aux_separation_coef', 0.0),
			'rgb_interventional_aux_separation_coef',
		),
		'ranking': _finite_nonnegative(
			_get(cfg, 'rgb_interventional_aux_ranking_coef', 0.0),
			'rgb_interventional_aux_ranking_coef',
		),
	}
	arm_by_coefs = {
		(0.0, 1.0, 0.0, 0.0): 'data_matched',
		(1.0, 0.0, 0.0, 0.0): 'background_only',
		(0.0, 1.0, 1.0, 1.0): 'real_fork_only',
		(1.0, 1.0, 1.0, 1.0): 'joint',
	}
	coef_tuple = tuple(coefs[name] for name in (
		'background', 'positive', 'separation', 'ranking'
	))
	if coef_tuple not in arm_by_coefs:
		raise ValueError(
			'RGB interventional coefficients must select exactly data-matched '
			'(0,1,0,0), background-only (1,0,0,0), fork-only (0,1,1,1), '
			'or joint (1,1,1,1).'
		)
	base_arm = arm_by_coefs[coef_tuple]
	pairing_contract = rgb_interventional_pairing.config(
		cfg, base_arm=base_arm, enabled=True,
	)
	arm = pairing_contract['arm']

	threshold = _finite_nonnegative(
		_get(cfg, 'rgb_interventional_aux_outcome_gap_threshold', 0.05),
		'rgb_interventional_aux_outcome_gap_threshold',
	)
	margin_min = _finite_nonnegative(
		_get(cfg, 'rgb_interventional_aux_margin_min', 0.05),
		'rgb_interventional_aux_margin_min',
	)
	margin_max = _finite_nonnegative(
		_get(cfg, 'rgb_interventional_aux_margin_max', 0.2),
		'rgb_interventional_aux_margin_max',
	)
	if threshold != 0.05 or margin_min != 0.05 or margin_max != 0.2:
		raise ValueError('RGB interventional outcome and margin constants drifted.')
	if margin_max < margin_min or margin_max > 1.0:
		raise ValueError('RGB interventional margins are invalid.')

	batch_size = _strict_positive_int(
		_get(cfg, 'rgb_interventional_aux_batch_size', 8),
		'rgb_interventional_aux_batch_size',
	)
	horizon = _strict_positive_int(
		_get(cfg, 'rgb_interventional_aux_horizon', 3),
		'rgb_interventional_aux_horizon',
	)
	frequency = _strict_positive_int(
		_get(cfg, 'rgb_interventional_aux_update_frequency', 1),
		'rgb_interventional_aux_update_frequency',
	)
	seed_offset = _strict_positive_int(
		_get(cfg, 'rgb_interventional_aux_seed_offset', 32452843),
		'rgb_interventional_aux_seed_offset',
	)
	if (
		batch_size != 8 or horizon != 3 or horizon != int(_get(cfg, 'horizon', -1))
		or frequency != 1 or seed_offset != 32452843
	):
		raise ValueError('RGB interventional sampler constants drifted from the screen.')
	if horizon > MAX_HORIZON:
		raise ValueError('RGB interventional horizon exceeds the supported maximum.')

	manifest_sha = _get(cfg, 'rgb_interventional_aux_manifest_sha256', None)
	capsule_format = _get(cfg, 'rgb_interventional_aux_capsule_format', None)
	source_sha = _get(cfg, 'rgb_interventional_aux_source_dataset_sha256', None)
	for value, name in (
		(manifest_sha, 'manifest SHA-256'), (source_sha, 'source SHA-256')
	):
		if value is not None:
			_require(
				isinstance(value, str) and len(value) == 64
				and all(char in '0123456789abcdef' for char in value),
				f'Bound RGB interventional {name} is malformed.',
			)
	if capsule_format is not None:
		_require(capsule_format == FORMAT, 'Bound RGB capsule format changed.')
	if require_bound_manifest and any(
		value is None for value in (manifest_sha, capsule_format, source_sha)
	):
		raise ValueError('RGB interventional training requires a bound capsule identity.')

	return {
		'format': 'rgb_interventional_auxiliary_contract_v1',
		'arm': arm,
		'task': str(_get(cfg, 'task', '')),
		'action_dim': int(_get(cfg, 'action_dim', 0)),
		'conditions': list(CONDITIONS),
		'background_coef': coefs['background'],
		'positive_coef': coefs['positive'],
		'separation_coef': coefs['separation'],
		'ranking_coef': coefs['ranking'],
		'outcome_gap_threshold': threshold,
		'margin_min': margin_min,
		'margin_max': margin_max,
		'batch_size': batch_size,
		'horizon': horizon,
		'update_frequency': frequency,
		'seed_offset': seed_offset,
		'augmentation_seed_offset': seed_offset + 1,
		'pairing': pairing_contract,
		'manifest_sha256': manifest_sha,
		'capsule_format': capsule_format,
		'source_dataset_manifest_sha256': source_sha,
		'online_observation_changed': False,
		'test_time_inputs': ['rgb'],
		'cutie_inputs': False,
		'oracle_model_inputs': False,
		'real_executed_siblings_only': pairing_contract['fork_correct'],
		'outcome_grounded_eligibility': (
			base_arm != 'data_matched' and pairing_contract['fork_correct']
		),
		'pair_targets_used_for_gradient': base_arm != 'data_matched',
		'background_twin_identity_used_for_gradient': arm in {
			'background_only', 'joint', 'shuffled_fork',
		} and pairing_contract['background_correct'],
		'data_matched_transition_streams': (
			[
				'clean_positive', 'hard_positive',
			]
			if arm == 'data_matched' else []
		),
		'shared_forward_contract': SHARED_FORWARD_CONTRACT,
		'forward_tensor_streams': [
			'clean_root', 'hard_root',
			'clean_positive', 'hard_positive',
			'clean_negative', 'hard_negative',
		],
		'forward_action_streams': ['clean_positive', 'hard_positive'],
		'augmentation_contract': (
			'one_shared_B_crop_for_roots_then_one_shared_B_crop_per_horizon_'
			'across_all_four_future_renders_v1'
		),
		'sampler': 'balanced_axis_relation_shuffled_bag_v1',
	}


def validate_checkpoint_record(
	record, expected_contract: dict | None,
) -> dict[str, int | str]:
	if expected_contract is None:
		if record is not None:
			raise RuntimeError('RGB interventional checkpoint cannot load when disabled.')
		return {'attempts': 0, 'successful_updates': 0, 'sampled_root_pairs': 0}
	source_contract = record.get('contract') if isinstance(record, dict) else None
	legacy_expected = dict(expected_contract)
	legacy_pairing = legacy_expected.pop('pairing', None)
	is_legacy_correct_pairing = bool(
		isinstance(legacy_pairing, dict)
		and legacy_pairing.get('mode') == 'correct'
		and source_contract == legacy_expected
	)
	if (
		not isinstance(record, dict) or set(record) != {'contract', 'supervision'}
		or (
			source_contract != expected_contract
			and not is_legacy_correct_pairing
		)
	):
		raise RuntimeError(
			'RGB interventional checkpoint contract mismatch. '
			f'record_keys={sorted(record) if isinstance(record, dict) else type(record)}; '
			f'contract_equal={source_contract == expected_contract}; '
			f'legacy_pairing={is_legacy_correct_pairing}.'
		)
	supervision = record.get('supervision')
	if not isinstance(supervision, dict):
		raise RuntimeError('RGB interventional checkpoint supervision is malformed.')
	values = {}
	for key in ('attempts', 'successful_updates', 'sampled_root_pairs'):
		value = supervision.get(key)
		if not isinstance(value, int) or isinstance(value, bool) or value < 1:
			raise RuntimeError(f'RGB interventional checkpoint {key} is invalid.')
		values[key] = int(value)
	if values['successful_updates'] != values['attempts']:
		raise RuntimeError('Every RGB interventional attempt must update once.')
	expected_samples = values['successful_updates'] * int(expected_contract['batch_size'])
	if values['sampled_root_pairs'] != expected_samples:
		raise RuntimeError('RGB interventional sampled pair count is inconsistent.')
	metrics = supervision.get('sampler_metrics')
	if not isinstance(metrics, dict) or any((
		metrics.get('sample_calls') != values['successful_updates'],
		metrics.get('sampled_root_pairs') != expected_samples,
		not isinstance(metrics.get('root_count'), int),
		metrics.get('root_count', 0) < 1,
	)):
		raise RuntimeError('RGB interventional sampler metrics are malformed.')
	trace = metrics.get('sample_sequence_sha256')
	if (
		not isinstance(trace, str) or len(trace) != 64
		or any(char not in '0123456789abcdef' for char in trace)
	):
		raise RuntimeError('RGB interventional sample-sequence trace is malformed.')
	values['sample_sequence_sha256'] = trace
	augmentation = supervision.get('augmentation_metrics')
	expected_crop_calls = values['successful_updates'] * (
		int(expected_contract['horizon']) + 1
	)
	expected_crop_rows = expected_crop_calls * int(expected_contract['batch_size'])
	if (
		not isinstance(augmentation, dict)
		or set(augmentation) != {
			'crop_calls', 'crop_rows', 'draw_shape_sequence_sha256',
			'generator_state_sha256',
		}
		or augmentation.get('crop_calls') != expected_crop_calls
		or augmentation.get('crop_rows') != expected_crop_rows
		or any(
			not isinstance(augmentation.get(key), str)
			or len(augmentation[key]) != 64
			or any(char not in '0123456789abcdef' for char in augmentation[key])
			for key in (
				'draw_shape_sequence_sha256', 'generator_state_sha256',
			)
		)
	):
		raise RuntimeError('RGB interventional augmentation trace is malformed.')
	values['augmentation_provenance'] = dict(augmentation)
	pairing_metrics = supervision.get('pairing_metrics')
	if is_legacy_correct_pairing and pairing_metrics is None:
		values['pairing_provenance'] = None
	else:
		rgb_interventional_pairing.validate_metrics(
			pairing_metrics,
			contract=expected_contract['pairing'],
			updates=values['successful_updates'],
		)
		values['pairing_provenance'] = dict(pairing_metrics)
	return values


def _safe_asset_path(root: Path, relative: object) -> Path:
	if not isinstance(relative, str) or not relative:
		raise ValueError('RGB capsule asset path is malformed.')
	path = Path(relative)
	if path.is_absolute() or '..' in path.parts:
		raise ValueError('RGB capsule asset path must stay relative to its root.')
	resolved = (root / path).resolve()
	try:
		resolved.relative_to(root.resolve())
	except ValueError as exc:
		raise ValueError('RGB capsule asset escaped its root.') from exc
	if resolved.is_symlink():
		raise ValueError('Symlinked RGB capsule assets are forbidden.')
	return resolved


def _require_array(arrays, name, shape, dtype):
	if name not in arrays:
		raise ValueError(f'RGB capsule is missing {name!r}.')
	value = np.asarray(arrays[name])
	if value.shape != shape or value.dtype != np.dtype(dtype):
		raise ValueError(
			f'{name} must be {np.dtype(dtype)} {shape}, got {value.dtype} {value.shape}.'
		)
	if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
		raise ValueError(f'{name} contains non-finite values.')
	return value


def validate_shard(
	arrays: Mapping[str, np.ndarray], *, action_dim: int, horizon: int,
	branch_magnitude: float, outcome_gap_threshold: float,
) -> dict[str, int]:
	"""Reject object/oracle leakage and invalid action/outcome supervision."""
	if not isinstance(arrays, Mapping) or set(arrays) != SHARD_KEYS:
		actual = sorted(arrays) if isinstance(arrays, Mapping) else type(arrays)
		raise ValueError(
			f'RGB capsule arrays must be exactly {sorted(SHARD_KEYS)!r}, got {actual!r}.'
		)
	_strict_positive_int(action_dim, 'action_dim')
	_strict_positive_int(horizon, 'horizon')
	if horizon > MAX_HORIZON:
		raise ValueError('RGB capsule horizon exceeds the supported maximum.')
	if not math.isfinite(float(branch_magnitude)) or not 0.0 < float(branch_magnitude) <= 1.0:
		raise ValueError('RGB branch magnitude must lie in (0,1].')
	if float(outcome_gap_threshold) != 0.05:
		raise ValueError('RGB outcome-gap threshold changed.')
	branches = 1 + 2 * action_dim
	for condition in CONDITIONS:
		_require_array(
			arrays, f'{condition}__root_rgb', (9, 64, 64), np.uint8,
		)
		_require_array(
			arrays, f'{condition}__future_rgb',
			(branches, horizon, 9, 64, 64), np.uint8,
		)
	actions = _require_array(
		arrays, 'branch__action', (branches, horizon, action_dim), np.float32,
	)
	codes = _require_array(arrays, 'branch__code', (branches,), np.int32)
	if not np.array_equal(codes, branch_codes(action_dim)):
		raise ValueError('RGB sibling branch codes changed order.')
	expected = np.zeros((branches, action_dim), dtype=np.float32)
	row = 1
	for axis in range(action_dim):
		expected[row, axis] = np.float32(branch_magnitude)
		expected[row + 1, axis] = np.float32(-branch_magnitude)
		row += 2
	if not np.array_equal(actions[:, 0], expected):
		raise ValueError('RGB interventions are not zero plus/minus every basis action.')
	if horizon > 1 and not np.array_equal(
		actions[:, 1:], np.broadcast_to(actions[0:1, 1:], actions[:, 1:].shape)
	):
		raise ValueError('RGB sibling continuation actions are not shared.')
	if np.any(actions < -1.0) or np.any(actions > 1.0):
		raise ValueError('RGB sibling actions left [-1,1].')

	gap = _require_array(
		arrays, 'pair__outcome_gap', (branches, branches, horizon), np.float32,
	)
	eligible = _require_array(
		arrays, 'pair__eligible', (branches, branches, horizon), np.bool_,
	)
	if np.any(gap < 0.0) or not np.allclose(gap, gap.swapaxes(0, 1), atol=1e-6):
		raise ValueError('RGB outcome gaps must be finite, non-negative, and symmetric.')
	indices = np.arange(branches)
	if not np.array_equal(gap[indices, indices], np.zeros((branches, horizon), np.float32)):
		raise ValueError('RGB outcome-gap diagonal must be exact zero.')
	off_diagonal = ~np.eye(branches, dtype=np.bool_)[:, :, None]
	expected_eligible = off_diagonal & (gap > np.float32(outcome_gap_threshold))
	if not np.array_equal(eligible, expected_eligible):
		raise ValueError('RGB pair eligibility does not match the outcome-gap contract.')
	return {
		'branches': branches,
		'horizon': horizon,
		'eligible_frames': int(eligible.sum()),
	}


def validate_manifest(path: Path, cfg=None) -> dict[str, Any]:
	path = Path(path).resolve()
	if not path.is_file() or path.is_symlink():
		raise FileNotFoundError(path)
	payload = json.loads(path.read_text(encoding='utf-8'))
	_require(isinstance(payload, dict), 'RGB capsule manifest must be an object.')
	_require(payload.get('format') == FORMAT, 'RGB capsule format mismatch.')
	_require(payload.get('status') == STATUS, 'RGB capsule is incomplete.')
	_require(
		payload.get('controller_auxiliary_training_authorized') is True,
		'RGB capsule lacks explicit auxiliary-training authorization.',
	)
	_require(payload.get('source_split') == 'train', 'RGB capsule must contain train roots.')
	_require(payload.get('model_input_only') is True, 'RGB capsule is not model-input-only.')
	_require(payload.get('test_time_input') == 'rgb_only', 'RGB test-time input changed.')
	_require(payload.get('oracle_arrays_present') is False, 'RGB capsule declares oracle arrays.')
	_require(payload.get('cutie_arrays_present') is False, 'RGB capsule declares Cutie arrays.')
	_require(payload.get('no_cutie') is True, 'RGB capsule is not explicitly Cutie-free.')
	_require(payload.get('masks_present') is False, 'RGB capsule declares mask arrays.')
	_require(
		payload.get('raw_privileged_arrays_present') is False,
		'RGB capsule declares raw privileged arrays.',
	)
	_require(payload.get('object_arrays_present') is False, 'RGB capsule declares object arrays.')
	_require(payload.get('condition_pair') == list(CONDITIONS), 'RGB condition pair changed.')
	_require(payload.get('exact_physical_twins') is True, 'RGB clean/hard twins are not exact.')
	_require(
		payload.get('whole_root_families_only') is True,
		'RGB capsule does not contain whole action-sibling root families.',
	)
	_require(
		payload.get('fake_shuffled_action_futures') is False,
		'Fake RGB action/future pairing is forbidden.',
	)
	_require(
		payload.get('policy_input_keys') == list(POLICY_INPUT_KEYS),
		'RGB capsule policy-input allow-list changed.',
	)
	_require(
		payload.get('training_target_keys') == list(TRAINING_TARGET_KEYS),
		'RGB capsule training-target allow-list changed.',
	)
	outcome = payload.get('outcome_gap')
	_require(isinstance(outcome, dict), 'RGB outcome-gap metadata is missing.')
	_require(outcome.get('threshold') == 0.05, 'RGB outcome-gap threshold changed.')
	_require(
		outcome.get('normalization') == 'train_root_coordinate_population_std_then_rms',
		'RGB outcome-gap normalization changed.',
	)
	_require(
		outcome.get('exact_replay_noise_floor') == 0.0,
		'RGB source replay was not exact.',
	)
	_require(outcome.get('ineligible_margin') == 0.0, 'Ineligible RGB pairs gained a margin.')
	background_contract = payload.get('background_twin_contract')
	_require(
		isinstance(background_contract, dict)
		and background_contract.get('exact_physics_per_branch') is True,
		'RGB background twins lack exact per-branch physics proof.',
	)
	action_contract = payload.get('action_sibling_contract')
	_require(
		isinstance(action_contract, dict)
		and action_contract.get('common_exact_physical_root') is True,
		'RGB action siblings lack a common exact physical root.',
	)
	source = payload.get('source_dataset')
	_require(isinstance(source, dict), 'RGB source dataset identity is missing.')
	_require(
		isinstance(source.get('format'), str) and source.get('format')
		and isinstance(source.get('manifest_sha256'), str)
		and len(source['manifest_sha256']) == 64
		and source.get('controller_training_authorized') is False
		and source.get('validation_performed_before_privilege_stripping') is True,
		'RGB source dataset identity is malformed.',
	)
	_require(
		source.get('validation_performed_before_derivation_and_stripping') is True,
		'RGB source was not validated before derived-target materialization.',
	)
	target_contract = payload.get('outcome_target_contract')
	_require(
		isinstance(target_contract, dict)
		and target_contract.get('source') == 'oracle_official_state_train_roots_only'
		and target_contract.get('raw_source_stripped_after_derivation') is True
		and target_contract.get('coordinate_scale') == 'population_std_all_train_branch_futures'
		and target_contract.get('distance') == 'coordinate_standardized_rms'
		and target_contract.get('symmetric') is True
		and target_contract.get('diagonal_zero') is True
		and target_contract.get('scale_epsilon') == 1e-6
		and target_contract.get('eligibility_threshold') == 0.05
		and target_contract.get('exact_replay_noise_floor') == 0.0
		and target_contract.get('eligibility')
		== 'off_diagonal_and_gap_strictly_greater_than_threshold'
		and isinstance(target_contract.get('coordinate_scale_sha256'), str)
		and len(target_contract['coordinate_scale_sha256']) == 64
		and isinstance(target_contract.get('coordinate_count'), int)
		and not isinstance(target_contract.get('coordinate_count'), bool)
		and target_contract['coordinate_count'] > 0
		and isinstance(target_contract.get('state_dim'), int)
		and not isinstance(target_contract.get('state_dim'), bool)
		and target_contract['state_dim'] > 0,
		'RGB outcome-target provenance is malformed.',
	)
	action_dim = payload.get('action_dim')
	horizon = payload.get('horizon')
	branch_magnitude = payload.get('branch_magnitude')
	_strict_positive_int(action_dim, 'action_dim')
	_strict_positive_int(horizon, 'horizon')
	groups = payload.get('groups')
	_require(isinstance(groups, list) and groups, 'RGB capsule has no root groups.')
	root_ids = payload.get('source_root_ids')
	_require(
		isinstance(root_ids, list) and len(root_ids) == len(groups),
		'RGB capsule source-root identity list is malformed.',
	)
	root = path.parent
	seen = set()
	validated = []
	for expected_root_id, record in zip(root_ids, groups):
		_require(isinstance(record, dict), 'RGB capsule group record is malformed.')
		root_id = record.get('root_id')
		_require(
			isinstance(root_id, int) and not isinstance(root_id, bool)
			and root_id >= 0 and root_id not in seen
			and expected_root_id == root_id,
			f'Invalid or duplicate RGB root id {root_id!r}.',
		)
		seen.add(root_id)
		source_group_sha = record.get('source_group_sha256')
		_require(
			isinstance(source_group_sha, str) and len(source_group_sha) == 64
			and all(char in '0123456789abcdef' for char in source_group_sha),
			f'RGB root {root_id} lost its source-group identity.',
		)
		asset = _safe_asset_path(root, record.get('relative_path'))
		_require(asset.is_file(), f'RGB capsule shard is missing: {asset}')
		expected_sha = record.get('sha256')
		_require(isinstance(expected_sha, str) and len(expected_sha) == 64, 'RGB shard SHA is missing.')
		_require(file_sha256(asset) == expected_sha, f'RGB root {root_id} SHA mismatch.')
		with np.load(asset, allow_pickle=False) as archive:
			arrays = {key: archive[key] for key in archive.files}
		validate_shard(
			arrays, action_dim=action_dim, horizon=horizon,
			branch_magnitude=float(branch_magnitude),
			outcome_gap_threshold=float(outcome['threshold']),
		)
		validated.append((root_id, asset))
	if cfg is not None:
		_require(payload.get('task') == _get(cfg, 'task'), 'RGB capsule task mismatch.')
		_require(action_dim == int(_get(cfg, 'action_dim', 0)), 'RGB action width mismatch.')
		requested_horizon = int(_get(cfg, 'rgb_interventional_aux_horizon', 3))
		_require(horizon >= requested_horizon, 'RGB capsule is shorter than requested.')
		bindings = (
			(_get(cfg, 'rgb_interventional_aux_manifest_sha256', None), file_sha256(path), 'manifest'),
			(_get(cfg, 'rgb_interventional_aux_capsule_format', None), FORMAT, 'format'),
			(_get(cfg, 'rgb_interventional_aux_source_dataset_sha256', None), source['manifest_sha256'], 'source'),
		)
		for bound, actual, name in bindings:
			if bound is not None:
				_require(bound == actual, f'Bound RGB {name} identity mismatch.')
	payload['_validated_groups'] = tuple(validated)
	payload['_manifest_sha256'] = file_sha256(path)
	return payload


@dataclass(frozen=True)
class InterventionalBatch:
	clean_root: torch.Tensor
	hard_root: torch.Tensor
	clean_positive_future: torch.Tensor
	hard_positive_future: torch.Tensor
	clean_negative_future: torch.Tensor
	hard_negative_future: torch.Tensor
	action: torch.Tensor
	negative_action: torch.Tensor
	outcome_gap: torch.Tensor | None
	eligible: torch.Tensor | None
	root_id: torch.Tensor
	positive_code: torch.Tensor
	negative_code: torch.Tensor


class Replay:
	"""Deterministic balanced sampler over immutable complete sibling roots."""

	def __init__(self, cfg, *, device):
		path = _get(cfg, 'rgb_interventional_aux_manifest', None)
		if path is None:
			raise ValueError('RGB interventional manifest is required for training.')
		self.manifest_path = Path(path).resolve()
		payload = validate_manifest(self.manifest_path, cfg)
		contract = validate_config(cfg, require_bound_manifest=True)
		if contract is None:
			raise RuntimeError('RGB replay cannot be constructed while disabled.')
		self._pair_targets_enabled = bool(
			contract['pair_targets_used_for_gradient']
		)
		self.manifest_sha256 = payload['_manifest_sha256']
		self.format = payload['format']
		self.task = payload['task']
		self.horizon = int(_get(cfg, 'rgb_interventional_aux_horizon', 3))
		self.batch_size = int(_get(cfg, 'rgb_interventional_aux_batch_size', 8))
		self.device = torch.device(device)
		self._rng = np.random.default_rng(
			int(_get(cfg, 'seed', 0))
			+ int(_get(cfg, 'rgb_interventional_aux_seed_offset', 32452843))
		)
		self._groups = []
		for root_id, asset in payload['_validated_groups']:
			with np.load(asset, allow_pickle=False) as archive:
				arrays = {key: np.ascontiguousarray(archive[key]) for key in SHARD_KEYS}
			self._groups.append((int(root_id), arrays))
		self._root_bag = np.empty(0, dtype=np.int64)
		self._pair_bag = np.empty((0, 2), dtype=np.int64)
		self._pair_population = balanced_ordered_code_pairs(
			int(_get(cfg, 'action_dim', 0))
		)
		self._calls = 0
		self._samples = 0
		self._eligible_frames = 0
		self._total_frames = 0
		# This digest is deliberately independent of the scientific arm.  Two
		# runs with the same capsule, seed and sampler contract must therefore
		# produce the same digest after every sample call.  It records identities,
		# not privileged outcome targets.
		self._sample_sequence = hashlib.sha256()
		self._sample_sequence.update(json.dumps({
			'format': 'rgb_interventional_sample_sequence_v1',
			'manifest_sha256': self.manifest_sha256,
			'seed': (
				int(_get(cfg, 'seed', 0))
				+ int(_get(cfg, 'rgb_interventional_aux_seed_offset', 32452843))
			),
			'batch_size': self.batch_size,
			'horizon': self.horizon,
			'pair_population': self._pair_population.tolist(),
		}, sort_keys=True, separators=(',', ':')).encode('utf-8'))

	@property
	def metrics(self) -> dict[str, int | str]:
		return {
			'sample_calls': int(self._calls),
			'sampled_root_pairs': int(self._samples),
			'root_count': len(self._groups),
			'eligible_pair_frames': int(self._eligible_frames),
			'sampled_pair_frames': int(self._total_frames),
			'sample_sequence_sha256': self._sample_sequence.hexdigest(),
		}

	def _draw(self, population, bag, count):
		values = []
		while len(values) < count:
			if len(bag) == 0:
				bag = population[self._rng.permutation(len(population))]
			take = min(count - len(values), len(bag))
			values.extend(bag[:take].tolist())
			bag = bag[take:]
		return np.asarray(values, dtype=np.int64), bag

	def _tensor(self, values):
		return torch.from_numpy(np.ascontiguousarray(values)).to(self.device).contiguous()

	def sample(self) -> InterventionalBatch:
		root_population = np.arange(len(self._groups), dtype=np.int64)
		root_index, self._root_bag = self._draw(
			root_population, self._root_bag, self.batch_size
		)
		pairs, self._pair_bag = self._draw(
			self._pair_population, self._pair_bag, self.batch_size
		)
		selected = [self._groups[int(index)] for index in root_index]
		codes = self._groups[0][1]['branch__code']
		code_to_index = {int(code): index for index, code in enumerate(codes)}
		positive = np.asarray([code_to_index[int(pair[0])] for pair in pairs], np.int64)
		negative = np.asarray([code_to_index[int(pair[1])] for pair in pairs], np.int64)
		self._sample_sequence.update(json.dumps({
			'call': self._calls,
			'root_id': [self._groups[int(index)][0] for index in root_index],
			'positive_code': pairs[:, 0].tolist(),
			'negative_code': pairs[:, 1].tolist(),
		}, sort_keys=True, separators=(',', ':')).encode('utf-8'))

		def roots(condition):
			return self._tensor(np.stack([
				arrays[f'{condition}__root_rgb'] for _, arrays in selected
			]))

		def futures(condition, indices):
			return self._tensor(np.stack([
				arrays[f'{condition}__future_rgb'][int(branch), :self.horizon]
				for (_, arrays), branch in zip(selected, indices)
			]).swapaxes(0, 1))

		action = self._tensor(np.stack([
			arrays['branch__action'][int(branch), :self.horizon]
			for (_, arrays), branch in zip(selected, positive)
		]).swapaxes(0, 1))
		negative_action = self._tensor(np.stack([
			arrays['branch__action'][int(branch), :self.horizon]
			for (_, arrays), branch in zip(selected, negative)
		]).swapaxes(0, 1))
		gap = None
		eligible = None
		if self._pair_targets_enabled:
			gap = self._tensor(np.stack([
				arrays['pair__outcome_gap'][int(left), int(right), :self.horizon]
				for (_, arrays), left, right in zip(selected, positive, negative)
			]).swapaxes(0, 1))
			eligible = self._tensor(np.stack([
				arrays['pair__eligible'][int(left), int(right), :self.horizon]
				for (_, arrays), left, right in zip(selected, positive, negative)
			]).swapaxes(0, 1))
		self._calls += 1
		self._samples += self.batch_size
		if eligible is not None:
			self._eligible_frames += int(eligible.sum().item())
			self._total_frames += int(eligible.numel())
		return InterventionalBatch(
			clean_root=roots('clean'), hard_root=roots('hard'),
			clean_positive_future=futures('clean', positive),
			hard_positive_future=futures('hard', positive),
			clean_negative_future=futures('clean', negative),
			hard_negative_future=futures('hard', negative),
			action=action, negative_action=negative_action,
			outcome_gap=gap, eligible=eligible,
			root_id=self._tensor(np.asarray([item[0] for item in selected], np.int64)),
			positive_code=self._tensor(pairs[:, 0].astype(np.int32)),
			negative_code=self._tensor(pairs[:, 1].astype(np.int32)),
		)


def transition_consistency_objective(
	predicted: torch.Tensor, target: torch.Tensor, *, rho: float,
) -> dict[str, torch.Tensor]:
	"""TD-MPC2 latent consistency over ordinary executed transitions only.

	This intentionally has no same-state twin, sibling, outcome-gap, eligibility,
	or ranking arguments.  The data-matched control can therefore consume the
	same RGB/action capsule without any intervention-derived target influencing
	its gradient.
	"""
	if predicted.ndim != 3 or predicted.shape != target.shape:
		raise ValueError(
			'Data-matched transition latents must share shape [H,B,D].'
		)
	if not predicted.is_floating_point() or not target.is_floating_point():
		raise ValueError('Data-matched transition latents must be floating point.')
	if not math.isfinite(float(rho)) or not 0.0 < float(rho) <= 1.0:
		raise ValueError('Data-matched transition rho must lie in (0,1].')
	frame_error = (predicted - target.detach()).square().mean(dim=-1)
	weights = predicted.new_tensor([
		float(rho) ** index for index in range(predicted.shape[0])
	]).unsqueeze(-1)
	loss = (frame_error * weights).sum() / float(
		predicted.shape[0] * predicted.shape[1]
	)
	return {'loss': loss, 'mse': frame_error.mean()}


def background_invariance_loss(clean: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
	"""Symmetric same-state latent regression; both RGB encodings receive gradient."""
	if clean.shape != hard.shape or clean.ndim < 2 or not clean.is_floating_point():
		raise ValueError('Clean/hard latent twins must share a floating-point shape.')
	return (clean - hard).square().mean()


def outcome_grounded_fork_objective(
	predicted: torch.Tensor,
	positive_target: torch.Tensor,
	negative_target: torch.Tensor,
	outcome_gap: torch.Tensor,
	eligible: torch.Tensor,
	*, rho: float, positive_coef: float, separation_coef: float,
	ranking_coef: float, margin_min: float, margin_max: float,
) -> dict[str, torch.Tensor]:
	"""Train dynamics and distinguish only demonstrably different outcomes.

	The positive transition target is stop-gradient, matching TD-MPC2's ordinary
	consistency target.  A separate separation term sends gradient into both
	executed sibling encodings, preventing an encoder collapse from making a
	learned-target margin vanish.  Eligibility is derived before privileged state
	is stripped; coincident outcomes receive exactly zero separation/ranking.
	"""
	if (
		predicted.ndim != 3 or predicted.shape != positive_target.shape
		or predicted.shape != negative_target.shape
	):
		raise ValueError('RGB fork latents must share shape [H,B,D].')
	if outcome_gap.shape != predicted.shape[:2] or eligible.shape != predicted.shape[:2]:
		raise ValueError('RGB outcome gap/eligibility must have shape [H,B].')
	if eligible.dtype != torch.bool:
		raise ValueError('RGB fork eligibility must be boolean.')
	if not 0.0 < float(rho) <= 1.0 or margin_max < margin_min:
		raise ValueError('RGB fork loss constants are invalid.')
	if any(value < 0.0 for value in (positive_coef, separation_coef, ranking_coef)):
		raise ValueError('RGB fork coefficients must be non-negative.')

	positive_error = (predicted - positive_target.detach()).square().mean(-1)
	negative_error = (predicted - negative_target.detach()).square().mean(-1)
	# Saturate normalized official-state gaps at one; this produces a bounded,
	# task-independent latent RMS margin in [0.05, 0.2].
	gap_scale = outcome_gap.detach().clamp(0.0, 1.0)
	margin = float(margin_min) + (
		float(margin_max) - float(margin_min)
	) * gap_scale
	target_distance = (
		positive_target - negative_target
	).square().mean(-1).clamp_min(1e-12).sqrt()
	separation_frame = F.relu(margin - target_distance).square()
	# Ranking compares squared errors, so use the squared RMS margin.
	ranking_frame = F.relu(positive_error - negative_error + margin.square())
	mask = eligible.to(predicted.dtype)
	weights = predicted.new_tensor([
		float(rho) ** index for index in range(predicted.shape[0])
	]).unsqueeze(-1)
	all_denominator = float(predicted.shape[0] * predicted.shape[1])
	eligible_denominator = (mask * weights).sum().clamp_min(1.0)
	positive_loss = (positive_error * weights).sum() / all_denominator
	separation_loss = (separation_frame * mask * weights).sum() / eligible_denominator
	ranking_loss = (ranking_frame * mask * weights).sum() / eligible_denominator
	weighted = (
		float(positive_coef) * positive_loss
		+ float(separation_coef) * separation_loss
		+ float(ranking_coef) * ranking_loss
	)
	return {
		'positive_loss': positive_loss,
		'separation_loss': separation_loss,
		'ranking_loss': ranking_loss,
		'weighted_loss': weighted,
		'positive_mse': positive_error.mean(),
		'wrong_sibling_mse': negative_error.mean(),
		'target_distance': target_distance.mean(),
		'eligible_rate': mask.mean(),
		'eligible_outcome_gap': (
			(outcome_gap * mask).sum() / mask.sum().clamp_min(1.0)
		),
		'ranking_accuracy': (
			((positive_error < negative_error).to(predicted.dtype) * mask).sum()
			/ mask.sum().clamp_min(1.0)
		),
		'margin_violation_rate': (
			((ranking_frame > 0).to(predicted.dtype) * mask).sum()
			/ mask.sum().clamp_min(1.0)
		),
	}

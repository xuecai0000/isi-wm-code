"""Held-out mechanism evaluation for pure-RGB interventional TD-MPC2.

This evaluator answers two deliberately narrow questions on the immutable
same-state branch dataset:

1. Do two renders of the *same physical state* map to nearby RGB latents?
2. When two genuinely executed actions have measurably different physical
   outcomes, does the checkpoint predict the matching RGB future more closely
   than the wrong action sibling?

The source dataset contains privileged simulator arrays so its collector can
prove exact clean/hard twins and real action interventions.  They are opened in
the scoring boundary only to reconstruct the train-fitted anonymous
outcome-gap eligibility mask.  The model-facing allow-list is smaller: raw RGB
is passed to the official TD-MPC2 encoder and recorded actions are passed to
``model.next``.  Object tensors, masks, Cutie features, physics, official
state, and condition identifiers can never enter the model.

The branch dataset does not contain counterfactual rewards.  Consequently the
reported quantity is named ``model_action_ranking_accuracy``; it is a latent
dynamics diagnostic and is never presented as MPC regret or control return.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for _path in (str(REPO_DIR), str(PROJECT_DIR)):
	while _path in sys.path:
		sys.path.remove(_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from common import rgb_interventional_auxiliary as auxiliary
from tools import collect_rof_real_action_branches as collector
from tools import materialize_rgb_interventional_training_capsule as materializer


FORMAT = 'rgb_interventional_checkpoint_evaluation_v1'
STATUS = 'rgb_interventional_checkpoint_evaluation_complete'
TASKS = materializer.SCREEN_TASKS
ARMS = (
	'tdmpc2', 'data_matched', 'background_only', 'fork_only', 'joint',
	'shuffled_background', 'shuffled_fork', 'shuffled_joint',
)
CONDITIONS = ('clean', 'hard')
HORIZONS = (1, 2, 3)
EXPECTED_TEST_ROOTS = 10
EXPECTED_TRAINING_STEPS = 300_000  # zc1 2026-09-19: long-budget campaigns run 300k steps
EXPECTED_TRAINING_EVAL_FREQUENCY = 10_000  # zc1 2026-09-20: long-budget diagnostic grid
EXPECTED_TRAINING_EVAL_EPISODES = 10
MODEL_OBSERVATION_INPUTS = ('rgb',)
MODEL_TRANSITION_INPUTS = ('latent', 'executed_action')

# These are the same explicit route closures written by the portable screen.
# Missing legacy keys do not prove a route was off, so a screen checkpoint must
# retain every one as a strict false value in its runtime snapshot.
FORBIDDEN_BOOLEAN_ROUTES = (
	'flat_anchor',
	'cutie_masked_rgb_enabled',
	'cutie_mask_guided_rgb_enabled',
	'robust_object_field_enabled',
	'rof_real_sibling_aux_enabled',
	'cutie_object_spatial_token_enabled',
	'cutie_object_variable_graph_enabled',
	'cutie_object_variable_graph_primary_skip_enabled',
	'cutie_object_true_entity_enabled',
	'cutie_object_belief_enabled',
	'cutie_object_belief_use_for_control',
	'cutie_object_last_valid_memory',
	'cutie_object_native_highres_enabled',
	'object_state_supervision_enabled',
	'object_state_supervision_collect_labels',
	'object_state_bottleneck_enabled',
	'cutie_object_allow_simulator_runtime',
	'cutie_object_allow_simulator_kinematics_runtime',
	'visual_pose_use_cutie_mask',
)
FORBIDDEN_IDENTITY_ROUTES = (
	'rof_real_sibling_aux_manifest',
	'rof_real_sibling_aux_manifest_sha256',
	'rof_real_sibling_aux_capsule_format',
	'rof_real_sibling_aux_source_dataset_sha256',
	'cutie_object_regression_encoder',
	'cutie_object_repo',
	'cutie_object_checkpoint',
	'cutie_object_support_path',
	'cutie_object_config_dir',
	'cutie_object_policy_burst_plan',
	'cutie_object_spatial_graph_path',
	'flat_anchor_support_path',
	'flat_anchor_impl_path',
	'flat_anchor_dino_repo',
	'flat_anchor_dino_checkpoint',
	'visual_pose_checkpoint',
)
ARM_COEFFICIENTS = {
	'tdmpc2': (0.0, 0.0, 0.0, 0.0),
	'data_matched': (0.0, 1.0, 0.0, 0.0),
	'background_only': (1.0, 0.0, 0.0, 0.0),
	'fork_only': (0.0, 1.0, 1.0, 1.0),
	'joint': (1.0, 1.0, 1.0, 1.0),
	'shuffled_background': (1.0, 1.0, 1.0, 1.0),
	'shuffled_fork': (1.0, 1.0, 1.0, 1.0),
	'shuffled_joint': (1.0, 1.0, 1.0, 1.0),
}
CONTRACT_ARM = {
	'data_matched': 'data_matched',
	'background_only': 'background_only',
	'fork_only': 'real_fork_only',
	'joint': 'joint',
	'shuffled_background': 'shuffled_background',
	'shuffled_fork': 'shuffled_fork',
	'shuffled_joint': 'shuffled_joint',
}


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
	payload = json.loads(Path(path).read_text(encoding='utf-8'))
	_require(isinstance(payload, dict), f'Expected a JSON object: {path}')
	return payload


def _is_sha256(value: Any) -> bool:
	return (
		isinstance(value, str) and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _safe_source_asset(manifest_path: Path, record: Mapping[str, Any]) -> Path:
	value = record.get('relative_path')
	_require(isinstance(value, str) and value, 'Source group path is missing.')
	relative = Path(value)
	_require(
		not relative.is_absolute() and '..' not in relative.parts,
		'Source group path must remain relative.',
	)
	root = manifest_path.parent.resolve()
	path = (root / relative).resolve()
	try:
		path.relative_to(root)
	except ValueError as exc:
		raise ValueError('Source group path escaped its manifest root.') from exc
	_require(path.is_file() and not path.is_symlink(), f'Missing source shard: {path}')
	_require(_sha256(path) == record.get('sha256'), f'Source shard identity changed: {path}')
	return path


def _source_records(source: Mapping[str, Any], split: str) -> list[Mapping[str, Any]]:
	groups = source.get('groups')
	_require(isinstance(groups, list), 'Source group table is malformed.')
	rows = [record for record in groups if record.get('split') == split]
	declared = source.get('root_splits', {}).get(split)
	_require(
		declared == [record.get('root_id') for record in rows],
		f'Source {split} roots disagree with root_splits.',
	)
	_require(rows, f'Source {split} split is empty.')
	return rows


def _fit_train_coordinate_scale(
	source_manifest: Path, source: Mapping[str, Any], capsule: Mapping[str, Any],
) -> np.ndarray:
	"""Reproduce and hash-check the materializer's train-only state scale."""
	branches = 1 + 2 * int(source['action_dim'])
	horizon = int(source['horizon'])
	state_dim = int(source['state_dim'])
	moments = materializer._OnlineCoordinateMoments()
	for record in _source_records(source, 'train'):
		asset = _safe_source_asset(source_manifest, record)
		with np.load(asset, allow_pickle=False) as archive:
			official = np.asarray(archive['oracle__official_state'])
		_require(
			official.shape == (branches, horizon + 1, state_dim)
			and official.dtype == np.float64
			and np.isfinite(official).all(),
			'Train-root official-state scoring tensor is malformed.',
		)
		moments.update(official[:, 1:, :].reshape(-1, state_dim))
	scale = moments.scale(epsilon=materializer.SCALE_EPSILON)
	target = capsule.get('outcome_target_contract', {})
	_require(
		target.get('coordinate_count') == moments.count
		and target.get('state_dim') == state_dim,
		'Training capsule scale dimensions disagree with the source train roots.',
	)
	observed = materializer.typed_array_sha256(
		'official_state_coordinate_scale', scale,
	)
	_require(
		observed == target.get('coordinate_scale_sha256'),
		'Train-fitted outcome coordinate scale does not match the capsule binding.',
	)
	return scale


def _validate_runtime_contract(
	raw: Mapping[str, Any], *, task: str, arm: str, action_dim: int,
	capsule_path: Path, capsule: Mapping[str, Any], expected_step: int,
	training_condition: str = 'clean',
) -> dict[str, Any] | None:
	_require(task in TASKS and arm in ARMS, 'Unknown RGB screen task/arm.')
	_require(training_condition in {'clean', 'hard'}, 'Unknown training condition.')
	expected = {
		'task': task,
		'obs': 'rgb',
		'obs_shape': {'rgb': [9, 64, 64]},
		'action_dim': action_dim,
		'multitask': False,
		'flat_anchor': False,
		'latent_dim': 512,
		'model_size': 5,
		'enc_dim': 256,
		'mlp_dim': 512,
		'num_enc_layers': 2,
		'steps': expected_step,
		'eval_freq': EXPECTED_TRAINING_EVAL_FREQUENCY,
		'eval_episodes': EXPECTED_TRAINING_EVAL_EPISODES,
		'episode_length': 500,
		'compile': False,
		'video_background_enabled': training_condition == 'hard',
		'video_background_split': 'train',
		'visual_foreground_erosion_pixels': 0,
	}
	for key, wanted in expected.items():
		_require(raw.get(key) == wanted, f'Runtime {key}={raw.get(key)!r}, expected {wanted!r}.')
	for key in FORBIDDEN_BOOLEAN_ROUTES:
		_require(raw.get(key) is False, f'Forbidden visual/state route is not false: {key}.')
	for key in FORBIDDEN_IDENTITY_ROUTES:
		_require(raw.get(key) is None, f'Forbidden visual/state identity is not null: {key}.')
	coefficients = tuple(float(raw.get(
		f'rgb_interventional_aux_{name}_coef', -1.0,
	)) for name in ('background', 'positive', 'separation', 'ranking'))
	_require(coefficients == ARM_COEFFICIENTS[arm], 'Runtime auxiliary coefficients select the wrong arm.')
	if arm == 'tdmpc2':
		_require(raw.get('rgb_interventional_aux_enabled') is False,
			'TD-MPC2 baseline unexpectedly enables RGB interventions.')
		for suffix in ('manifest', 'manifest_sha256', 'capsule_format', 'source_dataset_sha256'):
			_require(raw.get(f'rgb_interventional_aux_{suffix}') is None,
				'TD-MPC2 baseline carries auxiliary identity.')
		_require(auxiliary.validate_config(raw) is None,
			'Disabled auxiliary unexpectedly produced a contract.')
		return None

	_require(raw.get('rgb_interventional_aux_enabled') is True,
		'Interventional arm did not enable its training auxiliary.')
	contract = auxiliary.validate_config(raw, require_bound_manifest=True)
	_require(contract is not None and contract.get('arm') == CONTRACT_ARM[arm],
		'Runtime auxiliary contract names the wrong arm.')
	# Atomic publication renames ``OUTPUT.incomplete`` to ``OUTPUT`` after all
	# evaluations.  The training-time absolute path may therefore be stale in a
	# published runtime snapshot; immutable content identity, not location, is
	# the binding that must remain exact.
	declared_manifest = raw.get('rgb_interventional_aux_manifest')
	_require(isinstance(declared_manifest, str) and declared_manifest,
		'Runtime training-capsule path provenance is missing.')
	_require(raw.get('rgb_interventional_aux_manifest_sha256') == capsule['_manifest_sha256'],
		'Runtime training-capsule SHA-256 mismatch.')
	_require(raw.get('rgb_interventional_aux_capsule_format') == capsule['format'],
		'Runtime training-capsule format mismatch.')
	_require(
		raw.get('rgb_interventional_aux_source_dataset_sha256')
		== capsule['source_dataset']['manifest_sha256'],
		'Runtime source-dataset identity mismatch.',
	)
	return contract


def _checkpoint_record(
	checkpoint: Path, expected_contract: dict[str, Any] | None,
) -> dict[str, int]:
	import torch

	try:
		payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	except TypeError:  # PyTorch before the weights_only argument.
		payload = torch.load(checkpoint, map_location='cpu')
	_require(isinstance(payload, dict), 'Checkpoint payload is not a mapping.')
	contract = payload.get('checkpoint_contract')
	_require(
		isinstance(contract, dict)
		and contract.get('format') == 'tdmpc2_checkpoint_contract_v1'
		and contract.get('latent_dim') == 512,
		'Checkpoint base contract is malformed.',
	)
	for forbidden in (
		'robust_object_field', 'robust_object_field_auxiliary',
		'rof_real_sibling_auxiliary', 'mask_guided_rgb',
		'object_state_supervision', 'cutie_object_observation',
		'cutie_object_auxiliary', 'gt_articulated_pose_observation',
		'visual_articulated_pose_observation',
		'multimodal_articulated_pose_observation', 'cutie_proprio_observation',
	):
		_require(forbidden not in contract, f'Checkpoint retained forbidden contract: {forbidden}.')
	record = contract.get('rgb_interventional_auxiliary')
	return auxiliary.validate_checkpoint_record(record, expected_contract)


def _load_agent(
	*, task: str, raw: Mapping[str, Any], runtime_config: Path,
	checkpoint: Path, output: Path, training_condition: str = 'clean',
):
	import torch
	from isi_wm import TDMPC2
	from isi_wm.tools import evaluate_cutie_multitask_checkpoint as base

	_require(torch.cuda.is_available(), 'Checkpoint mechanism evaluation requires CUDA.')
	args = SimpleNamespace(
		task=task, backend='rgb', erosion_pixels=0,
		training_seed=int(raw['seed']), training_condition=training_condition,
		expected_training_steps=int(raw['steps']),
		expected_training_eval_freq=int(raw['eval_freq']),
		expected_training_eval_episodes=int(raw['eval_episodes']),
		env_seed=918273, background_seed=918277,
		condition='clean', episodes=20,
		checkpoint=checkpoint, runtime_config=runtime_config, output=output,
	)
	cfg = base._prepare(args, dict(raw))
	cfg.compile = False
	# zc1 2026-09-20: the pairing contract is derived from cfg.seed; it must
	# bind the training-time seed, not the evaluation env seed.
	cfg.seed = int(raw['seed'])
	agent = TDMPC2(cfg)
	agent.load(checkpoint)
	agent.eval()
	_require(set(agent.model._encoder) == {'rgb'},
		'Loaded checkpoint contains a non-RGB encoder route.')
	_require(not hasattr(agent.model, '_state_supervision_head'),
		'Loaded checkpoint contains a simulator-state decoder route.')
	return agent


def _encode_rgb(agent, rgb: np.ndarray, *, batch_size: int):
	"""Encode an RGB tensor through an explicit identity random-shift crop."""
	import torch

	value = np.asarray(rgb)
	_require(value.dtype == np.uint8 and value.shape[-3:] == (9, 64, 64),
		'Encoder input must be uint8 [...,9,64,64] RGB only.')
	leading = value.shape[:-3]
	flat = np.ascontiguousarray(value).reshape((-1, 9, 64, 64))
	encoder = agent.model._encoder['rgb']
	modules = list(encoder.children())
	_require(modules and modules[0].__class__.__name__ == 'ShiftAug',
		'Expected the official RGB encoder with ShiftAug first.')
	pad = int(modules[0].pad)
	rows = []
	with torch.no_grad():
		for start in range(0, len(flat), batch_size):
			stop = min(start + batch_size, len(flat))
			packet = torch.as_tensor(flat[start:stop], device=agent.device)
			# In ShiftAug coordinates, (pad,pad) is the exact identity crop.
			shift = torch.full(
				(stop - start, 1, 1, 2), pad,
				device=agent.device, dtype=torch.int64,
			)
			rows.append(agent._encode_rgb_interventional_shared(packet, shift))
	encoded = torch.cat(rows, dim=0)
	return encoded.reshape(leading + (encoded.shape[-1],))


def _metric_or_none(values: np.ndarray) -> float | None:
	array = np.asarray(values, dtype=np.float64).reshape(-1)
	if not len(array):
		return None
	_require(np.isfinite(array).all(), 'Mechanism metric contains a non-finite value.')
	return float(array.mean(dtype=np.float64))


def _comparison_samples(
	predicted: np.ndarray, target: np.ndarray, eligible: np.ndarray,
) -> dict[str, Any]:
	"""Return only outcome-eligible ordered sibling comparisons."""
	predicted = np.asarray(predicted, dtype=np.float64)
	target = np.asarray(target, dtype=np.float64)
	eligible = np.asarray(eligible, dtype=np.bool_)
	_require(
		predicted.ndim == 3 and predicted.shape == target.shape,
		'Predicted and target latent tensors must be [branch,horizon,latent].',
	)
	branches, horizon, _ = predicted.shape
	_require(eligible.shape == (branches, branches, horizon),
		'Eligibility must be [branch,branch,horizon].')
	_require(not np.any(eligible[np.arange(branches), np.arange(branches)]),
		'Eligibility diagonal must be false.')
	error = np.mean(
		(predicted[:, None, :, :] - target[None, :, :, :]) ** 2,
		axis=-1, dtype=np.float64,
	)
	correct = error[np.arange(branches), np.arange(branches)]
	correct_grid = np.broadcast_to(correct[:, None, :], error.shape)
	target_rms = np.sqrt(np.mean(
		(target[:, None, :, :] - target[None, :, :, :]) ** 2,
		axis=-1, dtype=np.float64,
	))
	predicted_rms = np.sqrt(np.mean(
		(predicted[:, None, :, :] - predicted[None, :, :, :]) ** 2,
		axis=-1, dtype=np.float64,
	))
	return {
		'correct': correct_grid[eligible],
		'wrong': error[eligible],
		'ranking': (correct_grid[eligible] < error[eligible]).astype(np.float64),
		'target_rms': target_rms[eligible],
		'predicted_rms': predicted_rms[eligible],
		# This denominator is independent of the eligibility threshold.  Keeping it
		# beside the selected samples makes a zero-selection result distinguishable
		# from a missing or malformed evaluation.
		'possible_comparisons': branches * (branches - 1) * horizon,
	}


def _comparison_summary(samples: Mapping[str, Any]) -> dict[str, Any]:
	count = int(np.asarray(samples['correct']).size)
	possible = samples.get('possible_comparisons')
	_require(
		isinstance(possible, (int, np.integer))
		and not isinstance(possible, (bool, np.bool_))
		and int(possible) > 0,
		'Mechanism comparison denominator is missing or invalid.',
	)
	possible = int(possible)
	_require(count <= possible, 'Eligible comparison count exceeds its denominator.')
	correct = _metric_or_none(samples['correct'])
	wrong = _metric_or_none(samples['wrong'])
	separation = None
	if correct is not None and wrong is not None and wrong > 0.0:
		separation = 1.0 - correct / wrong
	return {
		'eligible_comparisons': count,
		'possible_comparisons': possible,
		'eligible_comparison_coverage': float(count / possible),
		'metric_status': (
			'measured' if count else
			'not_applicable_no_eligible_comparisons'
		),
		'correct_action_prediction_mse': correct,
		'wrong_sibling_prediction_mse': wrong,
		'prediction_error_separation_fraction': separation,
		'model_action_ranking_accuracy': _metric_or_none(samples['ranking']),
		'encoded_target_action_separation_rms': _metric_or_none(samples['target_rms']),
		'predicted_action_separation_rms': _metric_or_none(samples['predicted_rms']),
	}


def _merge_samples(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
	_require(rows, 'Cannot aggregate an empty mechanism sample table.')
	merged = {
		key: np.concatenate([np.asarray(row[key]).reshape(-1) for row in rows])
		for key in ('correct', 'wrong', 'ranking', 'target_rms', 'predicted_rms')
	}
	possible = [row.get('possible_comparisons') for row in rows]
	_require(
		all(
			isinstance(value, (int, np.integer))
			and not isinstance(value, (bool, np.bool_))
			and int(value) > 0
			for value in possible
		),
		'Cannot aggregate a malformed mechanism comparison denominator.',
	)
	merged['possible_comparisons'] = sum(int(value) for value in possible)
	return merged


def _distance_rms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
	left = np.asarray(left, dtype=np.float64)
	right = np.asarray(right, dtype=np.float64)
	_require(left.shape == right.shape, 'Latent twin shapes differ.')
	return np.sqrt(np.mean((left - right) ** 2, axis=-1, dtype=np.float64))


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
	left = _metric_or_none(numerator)
	right = _metric_or_none(denominator)
	if left is None or right is None or right <= 0.0:
		return None
	return float(left / right)


@dataclass(frozen=True)
class _RootEvidence:
	root_id: int
	clean_root: np.ndarray
	hard_root: np.ndarray
	clean_future: np.ndarray
	hard_future: np.ndarray
	action: np.ndarray
	gap: np.ndarray
	eligible: np.ndarray


def _load_test_root(
	source_manifest: Path, record: Mapping[str, Any], *, scale: np.ndarray,
	action_dim: int, horizon: int,
) -> _RootEvidence:
	asset = _safe_source_asset(source_manifest, record)
	branches = 1 + 2 * action_dim
	with np.load(asset, allow_pickle=False) as archive:
		# Explicit model/scoring selections prevent the much wider source archive
		# from becoming a model packet by accident.
		clean_history = np.asarray(archive['clean__history__policy_rgb'])
		hard_history = np.asarray(archive['hard__history__policy_rgb'])
		clean_future = np.asarray(archive['clean__future__policy_rgb'])
		hard_future = np.asarray(archive['hard__future__policy_rgb'])
		action = np.asarray(archive['branch__action'])
		official_for_scoring_only = np.asarray(archive['oracle__official_state'])
	_require(
		clean_history.shape == hard_history.shape == (3, 9, 64, 64)
		and clean_history.dtype == hard_history.dtype == np.uint8,
		'Held-out RGB history schema changed.',
	)
	_require(
		clean_future.shape == hard_future.shape
		== (branches, int(collector.HORIZON), 9, 64, 64)
		and clean_future.dtype == hard_future.dtype == np.uint8,
		'Held-out RGB future schema changed.',
	)
	_require(
		action.shape == (branches, int(collector.HORIZON), action_dim)
		and action.dtype == np.float32,
		'Held-out executed-action schema changed.',
	)
	gap, eligible = materializer.derive_pair_targets(
		official_for_scoring_only, scale,
		threshold=materializer.OUTCOME_GAP_THRESHOLD,
	)
	return _RootEvidence(
		root_id=int(record['root_id']),
		clean_root=np.ascontiguousarray(clean_history[-1]),
		hard_root=np.ascontiguousarray(hard_history[-1]),
		clean_future=np.ascontiguousarray(clean_future[:, :horizon]),
		hard_future=np.ascontiguousarray(hard_future[:, :horizon]),
		action=np.ascontiguousarray(action[:, :horizon]),
		gap=np.ascontiguousarray(gap[:, :, :horizon]),
		eligible=np.ascontiguousarray(eligible[:, :, :horizon]),
	)


def _rollout(agent, root_z, action: np.ndarray):
	import torch

	actions = torch.as_tensor(action, device=agent.device, dtype=torch.float32)
	branches, horizon, _ = actions.shape
	z = root_z.expand(branches, -1).contiguous()
	rows = []
	with torch.no_grad():
		for index in range(horizon):
			z = agent.model.next(z, actions[:, index], None)
			rows.append(z)
	return torch.stack(rows, dim=1)


def _evaluate_root(agent, evidence: _RootEvidence, *, batch_size: int):
	clean_root_z = _encode_rgb(agent, evidence.clean_root[None], batch_size=batch_size)
	hard_root_z = _encode_rgb(agent, evidence.hard_root[None], batch_size=batch_size)
	clean_target = _encode_rgb(agent, evidence.clean_future, batch_size=batch_size)
	hard_target = _encode_rgb(agent, evidence.hard_future, batch_size=batch_size)
	clean_prediction = _rollout(agent, clean_root_z, evidence.action)
	hard_prediction = _rollout(agent, hard_root_z, evidence.action)

	to_numpy = lambda value: value.detach().cpu().numpy().astype(np.float32)
	root_background = _distance_rms(to_numpy(clean_root_z), to_numpy(hard_root_z))
	clean_target_np = to_numpy(clean_target)
	hard_target_np = to_numpy(hard_target)
	future_background = _distance_rms(clean_target_np, hard_target_np)
	condition_samples = {}
	condition_public = {}
	for condition, prediction, target in (
		('clean', to_numpy(clean_prediction), clean_target_np),
		('hard', to_numpy(hard_prediction), hard_target_np),
	):
		all_samples = _comparison_samples(prediction, target, evidence.eligible)
		by_horizon_samples = {}
		by_horizon = {}
		for index in range(evidence.action.shape[1]):
			selected = _comparison_samples(
				prediction[:, index:index + 1], target[:, index:index + 1],
				evidence.eligible[:, :, index:index + 1],
			)
			by_horizon_samples[str(index + 1)] = selected
			by_horizon[str(index + 1)] = _comparison_summary(selected)
		condition_samples[condition] = {
			'all': all_samples, 'by_horizon': by_horizon_samples,
		}
		condition_public[condition] = {
			'all_horizons': _comparison_summary(all_samples),
			'by_horizon': by_horizon,
		}

	# Normalize background drift by the encoded distance between only those
	# action siblings whose simulator outcomes passed the frozen eligibility gate.
	target_scale_all = np.concatenate([
		condition_samples[condition]['all']['target_rms']
		for condition in CONDITIONS
	])
	background_public = {
		'root_latent_rms': _metric_or_none(root_background),
		'future_latent_rms': _metric_or_none(future_background),
		'eligible_action_latent_rms': _metric_or_none(target_scale_all),
		'root_over_eligible_action_ratio': _ratio(root_background, target_scale_all),
		'future_over_eligible_action_ratio': _ratio(future_background, target_scale_all),
		'by_horizon': {},
	}
	for index in range(evidence.action.shape[1]):
		scale = np.concatenate([
			condition_samples[condition]['by_horizon'][str(index + 1)]['target_rms']
			for condition in CONDITIONS
		])
		background_public['by_horizon'][str(index + 1)] = {
			'future_latent_rms': _metric_or_none(future_background[:, index]),
			'eligible_action_latent_rms': _metric_or_none(scale),
			'future_over_eligible_action_ratio': _ratio(
				future_background[:, index], scale,
			),
		}
	return {
		'public': {
			'root_id': evidence.root_id,
			'eligible_pair_frames': int(evidence.eligible.sum()),
			'possible_pair_frames': int(
				evidence.eligible.shape[0]
				* (evidence.eligible.shape[0] - 1)
				* evidence.eligible.shape[2]
			),
			'eligible_pair_coverage': float(
				evidence.eligible.sum()
				/ (
					evidence.eligible.shape[0]
					* (evidence.eligible.shape[0] - 1)
					* evidence.eligible.shape[2]
				)
			),
			'metric_status': (
				'measured' if evidence.eligible.any() else
				'not_applicable_no_eligible_comparisons'
			),
			'background_invariance': background_public,
			'conditions': condition_public,
		},
		'samples': condition_samples,
		'root_background': root_background,
		'future_background': future_background,
	}


def _aggregate(rows: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
	conditions = {}
	for condition in CONDITIONS:
		all_samples = _merge_samples([
			row['samples'][condition]['all'] for row in rows
		])
		by_horizon = {}
		for index in range(horizon):
			by_horizon[str(index + 1)] = _comparison_summary(_merge_samples([
				row['samples'][condition]['by_horizon'][str(index + 1)]
				for row in rows
			]))
		conditions[condition] = {
			'all_horizons': _comparison_summary(all_samples),
			'by_horizon': by_horizon,
		}

	root_background = np.concatenate([row['root_background'].reshape(-1) for row in rows])
	future_background = np.concatenate([
		row['future_background'].reshape(-1) for row in rows
	])
	target_scale = np.concatenate([
		row['samples'][condition]['all']['target_rms']
		for row in rows for condition in CONDITIONS
	])
	background = {
		'root_latent_rms': _metric_or_none(root_background),
		'future_latent_rms': _metric_or_none(future_background),
		'eligible_action_latent_rms': _metric_or_none(target_scale),
		'root_over_eligible_action_ratio': _ratio(root_background, target_scale),
		'future_over_eligible_action_ratio': _ratio(future_background, target_scale),
		'by_horizon': {},
	}
	for index in range(horizon):
		future = np.concatenate([
			row['future_background'][:, index].reshape(-1) for row in rows
		])
		scale = np.concatenate([
			row['samples'][condition]['by_horizon'][str(index + 1)]['target_rms']
			for row in rows for condition in CONDITIONS
		])
		background['by_horizon'][str(index + 1)] = {
			'future_latent_rms': _metric_or_none(future),
			'eligible_action_latent_rms': _metric_or_none(scale),
			'future_over_eligible_action_ratio': _ratio(future, scale),
		}
	return {
		'background_invariance': background,
		'conditions': conditions,
	}


def evaluate(args) -> dict[str, Any]:
	import torch

	runtime_config = args.runtime_config.resolve()
	checkpoint = args.checkpoint.resolve()
	source_manifest = args.source_manifest.resolve()
	capsule_manifest = args.training_capsule_manifest.resolve()
	output = args.output.resolve()
	for path in (runtime_config, checkpoint, source_manifest, capsule_manifest):
		_require(path.is_file() and not path.is_symlink(), f'Missing/symlink input: {path}')
	_require(not output.exists(), f'Refusing to overwrite {output}.')
	_require(args.task in TASKS and args.arm in ARMS, 'Unknown screen task/arm.')
	_require(args.split == 'test', 'Only the held-out test split may be scored.')
	_require(args.disable_random_shift is True,
		'--disable-random-shift is mandatory for exact paired scoring.')
	_require(args.horizon == max(HORIZONS) == 3,
		'The registered mechanism horizon is exactly three.')
	_require(args.batch_size > 0, 'Encoding batch size must be positive.')
	_require(args.expected_training_step == EXPECTED_TRAINING_STEPS,
		'The registered checkpoint step is 300000.')

	collector.TASKS = TASKS
	collector.validate_dataset(source_manifest)
	source = _read_json(source_manifest)
	_require(source.get('task') == args.task, 'Source branch task mismatch.')
	_require(source.get('horizon') == int(collector.HORIZON),
		'Source branch horizon changed.')
	capsule = materializer.validate_capsule(capsule_manifest)
	_require(capsule.get('task') == args.task, 'Training capsule task mismatch.')
	_require(
		capsule['source_dataset']['manifest_sha256'] == _sha256(source_manifest),
		'Training capsule is bound to a different source dataset.',
	)
	_require(capsule.get('source_split') == 'train',
		'Training capsule must not contain held-out roots.')
	raw = _read_json(runtime_config)
	expected_contract = _validate_runtime_contract(
		raw, task=args.task, arm=args.arm,
		action_dim=int(source['action_dim']), capsule_path=capsule_manifest,
		capsule=capsule, expected_step=args.expected_training_step,
		training_condition=args.training_condition,
	)
	expected_checkpoint = runtime_config.parent / 'models' / 'final.pt'
	_require(checkpoint == expected_checkpoint.resolve(),
		'Only the runtime-paired final checkpoint is accepted.')
	supervision = _checkpoint_record(checkpoint, expected_contract)

	scale = _fit_train_coordinate_scale(source_manifest, source, capsule)
	test_records = _source_records(source, args.split)
	_require(len(test_records) == EXPECTED_TEST_ROOTS,
		f'Expected exactly {EXPECTED_TEST_ROOTS} complete held-out roots.')

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.set_float32_matmul_precision('high')
	agent = _load_agent(
		task=args.task, raw=raw, runtime_config=runtime_config,
		checkpoint=checkpoint, output=output,
		training_condition=args.training_condition,
	)
	evaluated = []
	for record in test_records:
		evidence = _load_test_root(
			source_manifest, record, scale=scale,
			action_dim=int(source['action_dim']), horizon=args.horizon,
		)
		evaluated.append(_evaluate_root(agent, evidence, batch_size=args.batch_size))
	aggregate = _aggregate(evaluated, args.horizon)
	return {
		'format': FORMAT,
		'status': STATUS,
		'task': args.task,
		'arm': args.arm,
		'training_condition': args.training_condition,
		'split': args.split,
		'root_count': len(evaluated),
		'horizons': list(HORIZONS),
		'random_shift': 'disabled_exact_identity_crop',
		'model_input_contract': {
			'observation_inputs': list(MODEL_OBSERVATION_INPUTS),
			'transition_inputs': list(MODEL_TRANSITION_INPUTS),
			'cutie': False,
			'object': False,
			'mask': False,
			'official_state': False,
			'physics_state': False,
			'proprioception': False,
			'condition_id': False,
		},
		'privileged_scoring_contract': {
			'official_state_use': (
				'train_coordinate_scale_and_test_outcome_eligibility_only'
			),
			'official_state_sent_to_model': False,
			'physics_state_sent_to_model': False,
			'objects_or_masks_sent_to_model': False,
			'eligibility_threshold': materializer.OUTCOME_GAP_THRESHOLD,
			'coordinate_scale_sha256': capsule[
				'outcome_target_contract'
			]['coordinate_scale_sha256'],
		},
		'scientific_scope': {
			'metric_name': 'model_action_ranking_accuracy',
			'planning_regret_reported': False,
			'reason': 'source_branches_do_not_store_counterfactual_reward',
			'no_eligible_comparison_semantics': (
				'numeric_mechanism_metrics_are_json_null_and_coverage_is_zero'
			),
		},
		'metrics': aggregate,
		'per_root': [row['public'] for row in evaluated],
		'provenance': {
			'runtime_config': str(runtime_config),
			'runtime_config_sha256': _sha256(runtime_config),
			'checkpoint': str(checkpoint),
			'checkpoint_sha256': _sha256(checkpoint),
			'checkpoint_step': args.expected_training_step,
			'checkpoint_auxiliary_supervision': supervision,
			'training_capsule_manifest': str(capsule_manifest),
			'training_capsule_manifest_sha256': _sha256(capsule_manifest),
			'source_manifest': str(source_manifest),
			'source_manifest_sha256': _sha256(source_manifest),
			'evaluator_sha256': _sha256(Path(__file__).resolve()),
		},
	}


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
	path = path.resolve()
	path.parent.mkdir(parents=True, exist_ok=True)
	_require(not path.exists(), f'Refusing to overwrite {path}.')
	temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
	try:
		with temporary.open('x', encoding='utf-8') as stream:
			json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
			stream.write('\n')
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--arm', choices=ARMS, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--source-manifest', type=Path, required=True)
	parser.add_argument('--training-capsule-manifest', type=Path, required=True)
	parser.add_argument('--split', choices=('test',), default='test')
	parser.add_argument('--expected-training-step', type=int, default=30_000)
	parser.add_argument(
		'--training-condition', choices=('clean', 'hard'), default='clean',
		help='Background condition used by the checkpoint online training stream.',
	)
	parser.add_argument('--horizon', type=int, default=3)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--disable-random-shift', action='store_true')
	parser.add_argument('--output', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	payload = evaluate(args)
	_atomic_write(args.output, payload)
	print('RGB_INTERVENTIONAL_CHECKPOINT_EVALUATION_COMPLETE', json.dumps({
		'task': payload['task'], 'arm': payload['arm'],
		'root_count': payload['root_count'],
		'output': str(args.output.resolve()),
	}, allow_nan=False), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

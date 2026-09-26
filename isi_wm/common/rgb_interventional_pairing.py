"""Deterministic pairing controls for the RGB interventional auxiliary.

The schedule is deliberately independent of both replay sampling and image
augmentation RNG.  Every mode draws the same two candidate derangements on
every update.  The correct-pair control applies identity indices, while the
three shuffled controls apply the relevant candidate.  Consequently a paired
correct/shuffled run with the same replay transcript has identical capsule
samples and augmentation draws; only the correspondence used by the losses
changes.

Derangements are checked against semantic sample identities rather than row
indices.  This matters when a replay minibatch crosses a shuffled-bag boundary
and can contain a repeated root or action code.  A shuffled background donor
must have a different root, and a shuffled fork donor must have a different
``(root, negative-action-code)`` tuple, so the correct target can never survive
accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np


FORMAT = 'rgb_interventional_pairing_schedule_v1'
SEED_OFFSET = 32452845
MODES = (
	'correct',
	'shuffle_background',
	'shuffle_fork',
	'shuffle_both',
)
MODE_TO_ARM = {
	'correct': 'joint',
	'shuffle_background': 'shuffled_background',
	'shuffle_fork': 'shuffled_fork',
	'shuffle_both': 'shuffled_joint',
}
ARM_TO_MODE = {arm: mode for mode, arm in MODE_TO_ARM.items()}


def _get(cfg, key: str, default=None):
	return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def config(cfg, *, base_arm: str, enabled: bool) -> dict:
	"""Validate pairing fields and return their checkpoint contract.

	Only the full joint objective may shuffle correspondences.  This keeps the
	existing background-only, fork-only and data-matched meanings unambiguous.
	Missing fields resolve to the legacy ``correct`` behavior.
	"""
	mode = _get(cfg, 'rgb_interventional_aux_pairing_mode', 'correct')
	_require(type(mode) is str and mode in MODES, (
		'rgb_interventional_aux_pairing_mode must be one of '
		f'{list(MODES)!r}.'
	))
	seed_offset = _get(
		cfg, 'rgb_interventional_aux_pairing_seed_offset', SEED_OFFSET,
	)
	_require(
		type(seed_offset) is int and seed_offset == SEED_OFFSET,
		'rgb_interventional_aux_pairing_seed_offset changed from the frozen value.',
	)
	if not enabled:
		_require(mode == 'correct', 'Shuffled RGB pairing requires the auxiliary.')
	if mode != 'correct':
		_require(
			base_arm == 'joint',
			'Shuffled RGB pairing is defined only for the full joint objective.',
		)
	runtime_seed = _get(cfg, 'seed', 0)
	batch_size = _get(cfg, 'rgb_interventional_aux_batch_size', 8)
	_require(type(runtime_seed) is int, 'RGB pairing runtime seed must be an integer.')
	_require(
		type(batch_size) is int and batch_size > 1,
		'RGB pairing batch size must exceed one.',
	)
	return {
		'format': FORMAT,
		'mode': mode,
		'seed_offset': seed_offset,
		'effective_seed': runtime_seed + seed_offset,
		'batch_size': batch_size,
		'arm': MODE_TO_ARM[mode] if base_arm == 'joint' else base_arm,
		'background_correct': mode not in {'shuffle_background', 'shuffle_both'},
		'fork_correct': mode not in {'shuffle_fork', 'shuffle_both'},
		'candidate_schedule': 'identity_aware_uniform_rejection_derangement_v1',
		'candidate_draws_identical_across_modes': True,
		'correct_pair_fixed_points_allowed_when_shuffled': 0,
	}


def _one_dimensional(values, name: str) -> np.ndarray:
	array = np.asarray(values)
	_require(array.ndim == 1 and array.size > 1, f'{name} must contain at least two rows.')
	_require(
		np.issubdtype(array.dtype, np.integer),
		f'{name} must contain integer identities.',
	)
	return np.ascontiguousarray(array.astype(np.int64, copy=False))


def _candidate(
	rng: np.random.Generator, keys: np.ndarray, *, label: str,
) -> np.ndarray:
	"""Draw a permutation with no semantically correct donor.

	Rejection sampling is intentional: it is simple, auditable, and both the
	correct and shuffled arms execute the exact same candidate-generation stream.
	The necessary condition below rejects impossible duplicate-heavy batches
	without looping.
	"""
	rows = int(keys.shape[0])
	if keys.ndim == 1:
		key_rows = [int(value) for value in keys]
	else:
		key_rows = [tuple(int(value) for value in row) for row in keys]
	counts = {}
	for key in key_rows:
		counts[key] = counts.get(key, 0) + 1
	_require(
		max(counts.values()) <= rows // 2,
		f'{label} minibatch has no correct-pair-free permutation.',
	)
	identity = np.arange(rows, dtype=np.int64)
	for _ in range(100_000):
		permutation = np.ascontiguousarray(rng.permutation(rows), dtype=np.int64)
		if all(key_rows[index] != key_rows[int(donor)] for index, donor in enumerate(permutation)):
			return permutation
	raise RuntimeError(f'Could not draw a {label} identity-aware derangement.')


def _canonical_update(hasher, payload: Mapping) -> None:
	hasher.update(json.dumps(
		payload, sort_keys=True, separators=(',', ':'), allow_nan=False,
	).encode('utf-8'))


@dataclass(frozen=True)
class PairingIndices:
	background: np.ndarray
	fork: np.ndarray
	background_correct_pairs: int
	fork_correct_pairs: int


class Schedule:
	"""Generate auditable correct or shuffled loss correspondences."""

	def __init__(self, *, seed: int, batch_size: int, mode: str):
		_require(type(seed) is int, 'Pairing seed must be an integer.')
		_require(type(batch_size) is int and batch_size > 1, 'Pairing batch size must exceed one.')
		_require(type(mode) is str and mode in MODES, f'Unknown pairing mode {mode!r}.')
		self.seed = seed
		self.batch_size = batch_size
		self.mode = mode
		self._rng = np.random.default_rng(seed)
		self._calls = 0
		self._background_correct = 0
		self._fork_correct = 0
		self._candidate_sequence = hashlib.sha256()
		self._applied_sequence = hashlib.sha256()
		header = {
			'format': FORMAT, 'seed': seed, 'batch_size': batch_size,
		}
		_canonical_update(self._candidate_sequence, header)
		_canonical_update(self._applied_sequence, {**header, 'mode': mode})

	def sample(self, *, root_id, positive_code, negative_code) -> PairingIndices:
		root = _one_dimensional(root_id, 'root_id')
		positive = _one_dimensional(positive_code, 'positive_code')
		negative = _one_dimensional(negative_code, 'negative_code')
		_require(
			len(root) == len(positive) == len(negative) == self.batch_size,
			'Pairing identities do not match the frozen batch size.',
		)
		# Root identity alone is sufficient to forbid every true clean/hard twin.
		background_candidate = _candidate(
			self._rng, root, label='background',
		)
		# A negative RGB target is uniquely identified for this purpose by its root
		# and executed negative action code. Positive code is hashed as provenance
		# but need not constrain the donor once the negative target is guaranteed
		# to differ.
		fork_keys = np.stack([root, negative], axis=1)
		fork_candidate = _candidate(self._rng, fork_keys, label='fork')
		identity = np.arange(self.batch_size, dtype=np.int64)
		background = (
			background_candidate
			if self.mode in {'shuffle_background', 'shuffle_both'} else identity
		)
		fork = (
			fork_candidate
			if self.mode in {'shuffle_fork', 'shuffle_both'} else identity
		)
		background_correct = int(np.sum(root == root[background]))
		fork_correct = int(np.sum(np.all(fork_keys == fork_keys[fork], axis=1)))
		if self.mode in {'shuffle_background', 'shuffle_both'}:
			_require(background_correct == 0, 'Shuffled background retained a correct twin.')
		if self.mode in {'shuffle_fork', 'shuffle_both'}:
			_require(fork_correct == 0, 'Shuffled fork retained a correct negative target.')
		candidate_payload = {
			'call': self._calls,
			'root_id': root.tolist(),
			'positive_code': positive.tolist(),
			'negative_code': negative.tolist(),
			'background_candidate': background_candidate.tolist(),
			'fork_candidate': fork_candidate.tolist(),
		}
		_canonical_update(self._candidate_sequence, candidate_payload)
		_canonical_update(self._applied_sequence, {
			'call': self._calls,
			'background': background.tolist(), 'fork': fork.tolist(),
		})
		self._calls += 1
		self._background_correct += background_correct
		self._fork_correct += fork_correct
		return PairingIndices(
			background=np.ascontiguousarray(background),
			fork=np.ascontiguousarray(fork),
			background_correct_pairs=background_correct,
			fork_correct_pairs=fork_correct,
		)

	@property
	def metrics(self) -> dict:
		rows = self._calls * self.batch_size
		return {
			'format': FORMAT,
			'mode': self.mode,
			'seed': self.seed,
			'batch_size': self.batch_size,
			'calls': self._calls,
			'rows': rows,
			'background_correct_pairs': self._background_correct,
			'fork_correct_pairs': self._fork_correct,
			'background_correct_rate': (
				self._background_correct / rows if rows else math.nan
			),
			'fork_correct_rate': self._fork_correct / rows if rows else math.nan,
			'candidate_sequence_sha256': self._candidate_sequence.hexdigest(),
			'applied_sequence_sha256': self._applied_sequence.hexdigest(),
		}


def validate_metrics(metrics, *, contract: Mapping, updates: int) -> None:
	"""Fail closed on checkpoint provenance for a joint pairing schedule."""
	_require(isinstance(metrics, Mapping), 'RGB pairing checkpoint metrics are missing.')
	_require(metrics.get('format') == FORMAT, 'RGB pairing metric format changed.')
	_require(metrics.get('mode') == contract.get('mode'), 'RGB pairing metric mode changed.')
	_require(metrics.get('seed') == contract.get('effective_seed'), 'RGB pairing seed changed.')
	_require(metrics.get('batch_size') == contract.get('batch_size'), 'RGB pairing batch changed.')
	_require(metrics.get('calls') == updates, 'RGB pairing/update counts differ.')
	rows = updates * int(contract['batch_size'])
	_require(metrics.get('rows') == rows, 'RGB pairing row count is inconsistent.')
	for key in ('candidate_sequence_sha256', 'applied_sequence_sha256'):
		value = metrics.get(key)
		_require(
			isinstance(value, str) and len(value) == 64
			and all(character in '0123456789abcdef' for character in value),
			f'RGB pairing {key} is malformed.',
		)
	expected_background = rows if contract['background_correct'] else 0
	expected_fork = rows if contract['fork_correct'] else 0
	_require(
		metrics.get('background_correct_pairs') == expected_background
		and metrics.get('fork_correct_pairs') == expected_fork,
		'RGB pairing retained or removed an unexpected correct correspondence.',
	)

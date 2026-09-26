import importlib.util
import pathlib
import random
import sys
from contextlib import contextmanager, nullcontext

import gymnasium as gym
import numpy as np
import torch


_MISSING = object()
_TEACHER_CONFIG_KEYS = (
	'support_path',
	'dino_model',
	'dino_repo',
	'dino_input_size',
	'color_window',
	'color_score_weight',
	'goal_dino_score_weight',
	'goal_color_score_weight',
	'role_contrast_weight',
	'structured_candidates',
	'structured_nms_radius',
	'structure_weight',
	'structure_length_tolerance',
	'temporal_beam_size',
	'temporal_unary_scale_floor',
	'temporal_position_scale',
	'temporal_position_weight',
	'temporal_velocity_scale',
	'temporal_velocity_weight',
	'temporal_elbow_weight',
)

_EVENT_CONFIG_DEFAULTS = {
	'enabled': False,
	'max_interval': 12,
	'min_refresh_gap': 4,
	'confidence_threshold': 0.30,
	'structure_tolerance': 0.20,
	'color_search_radius': 6,
	'color_confidence_threshold': 0.20,
}


@contextmanager
def _preserve_random_state(device):
	"""Keep construction of the frozen teacher outside the RL RNG stream."""
	python_state = random.getstate()
	numpy_state = np.random.get_state()
	devices = []
	if device.type == 'cuda':
		# A dependency may touch a CUDA device other than the requested one during
		# import or initialization, so preserve every visible logical device.
		devices = list(range(torch.cuda.device_count()))
	try:
		with torch.random.fork_rng(devices=devices, enabled=True):
			yield
	finally:
		random.setstate(python_state)
		np.random.set_state(numpy_state)


class _ConfigView(dict):
	"""Small dict-like config compatible with the legacy anchor teacher."""

	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError as exc:
			raise AttributeError(key) from exc


def _config_value(cfg, key, default=_MISSING):
	"""Read a top-level locator option, accepting an optional flat_anchor_ prefix."""
	for candidate in (f'flat_anchor_{key}', key):
		try:
			value = cfg.get(candidate, _MISSING)
		except (AttributeError, KeyError):
			value = getattr(cfg, candidate, _MISSING)
		if value is not _MISSING:
			return value
	if default is not _MISSING:
		return default
	raise KeyError(
		f'Missing FlatAnchor config value "{key}" '
		f'(also accepted as "flat_anchor_{key}").'
	)


def _teacher_config(cfg):
	values = {key: _config_value(cfg, key) for key in _TEACHER_CONFIG_KEYS}
	if not values['support_path']:
		raise ValueError(
			'flat_anchor_support_path must point to a manually annotated JSON file.'
		)
	values['allow_diagnostic_support'] = bool(
		_config_value(cfg, 'allow_diagnostic_support', False)
	)
	values['dino_checkpoint'] = _config_value(cfg, 'dino_checkpoint', None)
	values['confidence_scale'] = _config_value(cfg, 'confidence_scale', 2.0)
	return _ConfigView(values)


def _event_config(cfg):
	return _ConfigView({
		key: _config_value(cfg, f'event_{key}', default)
		for key, default in _EVENT_CONFIG_DEFAULTS.items()
	})


def _load_teacher_class(impl_path):
	path = pathlib.Path(str(impl_path)).expanduser().resolve()
	if not path.is_file():
		raise FileNotFoundError(f'FlatAnchor implementation not found: {path}')
	module_name = f'_tdmpc2_flat_anchor_{abs(hash(str(path)))}'
	spec = importlib.util.spec_from_file_location(module_name, path)
	if spec is None or spec.loader is None:
		raise ImportError(f'Unable to load FlatAnchor implementation: {path}')
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	try:
		return module.CausalAnchorStateTeacher
	except AttributeError as exc:
		raise ImportError(
			f'{path} does not define CausalAnchorStateTeacher'
		) from exc


class FlatAnchorWrapper(gym.Wrapper):
	"""Augment stacked RGB observations with a frozen 25-D anchor observation."""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.cfg = cfg
		self._device = torch.device(str(_config_value(cfg, 'device')))
		self._mode = str(_config_value(cfg, 'mode', 'residual'))
		self._object_graph = self._mode == 'object_graph'
		self._structured_graph = self._mode in {
			'object_graph', 'hybrid_graph', 'reward_graph'
		}
		# The locator model stays on CUDA, while its tiny temporal state and final
		# 25-D observation stay on CPU in object-only mode. This avoids several
		# needless device round-trips per environment step.
		self._state_device = torch.device('cpu') if self._structured_graph else self._device
		# The environment is created before TDMPC2. DINO construction initializes
		# many modules, so without isolation it changes the nominally matched agent
		# initialization and all subsequent stochastic sampling for the same seed.
		with _preserve_random_state(self._device):
			teacher_cls = _load_teacher_class(_config_value(cfg, 'impl_path'))
			base_teacher = teacher_cls(_teacher_config(cfg), device=self._device)
			event_cfg = _event_config(cfg)
			if self._structured_graph and bool(event_cfg.enabled):
				from envs.wrappers.event_anchor_state import (
					EventTriggeredAnchorStateTeacher,
				)
				self.teacher = EventTriggeredAnchorStateTeacher(
					base_teacher, event_cfg
				)
			else:
				self.teacher = base_teacher
		self._output_dim = int(getattr(self.teacher, 'OUTPUT_DIM', 25))
		if self._output_dim != 25:
			raise ValueError(
				f'FlatAnchor teacher must output 25 values, got {self._output_dim}'
			)
		spaces = {
			'anchor': gym.spaces.Box(
				low=-np.inf,
				high=np.inf,
				shape=(self._output_dim,),
				dtype=np.float32,
			),
		}
		if not self._object_graph:
			spaces = {'rgb': env.observation_space, **spaces}
		self.observation_space = gym.spaces.Dict(spaces)
		self._state = None

	def _initial_state(self):
		return self.teacher.initial_state(
			batch=1,
			beam_size=self.teacher.beam_size,
			device=self._state_device,
		)

	@staticmethod
	def _latest_image(obs):
		value = torch.as_tensor(obs)
		if value.ndim != 3 or value.shape[0] < 3:
			raise ValueError(
				'FlatAnchor expects stacked channel-first RGB observations; '
				f'got {tuple(value.shape)}'
			)
		return value[-3:].permute(1, 2, 0).contiguous().unsqueeze(0)

	def _augment(self, obs, is_first):
		image = self._latest_image(obs)
		first = torch.tensor([is_first], dtype=torch.bool, device=self._state_device)
		# Legacy modes retain exact RNG isolation. The event locator only executes
		# deterministic inference operations, so object-only mode avoids the costly
		# per-frame CUDA RNG snapshot/restore.
		context = nullcontext() if self._structured_graph else _preserve_random_state(self._device)
		with context:
			if is_first or self._state is None:
				self._state = self._initial_state()
			anchor, self._state = self.teacher.extract(
				image,
				self._state,
				first,
				output_device=self._state_device,
			)
		anchor = anchor.reshape(1, self._output_dim)[0].detach().cpu().float()
		if self._object_graph:
			return {'anchor': anchor}
		return {'rgb': obs, 'anchor': anchor}

	def reset(self):
		return self._augment(self.env.reset(), is_first=True)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return self._augment(obs, is_first=False), reward, done, info

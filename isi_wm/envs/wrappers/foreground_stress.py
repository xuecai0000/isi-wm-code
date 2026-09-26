"""Deterministic foreground-erosion stress wrapper for clean DMC pixels.

The wrapper operates only on the rendered observation returned by ``Pixels``.
It does not inspect or mutate simulator state.  A pixel is clean DMC background
when its blue channel is strictly greater than both red and green, matching the
background predicate used by :mod:`video_background`.  Foreground is eroded by
an integer Chebyshev radius of 0, 1, or 2 pixels independently in every RGB
frame of the causal stack.  Removed foreground pixels are filled with the
channel-wise median of that frame's clean blue background.

Run this file directly from the repository root for its local contract check::

    python tdmpc2/envs/wrappers/foreground_stress.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


_ALLOWED_EROSION_PIXELS = (0, 1, 2)


def _blue_background_mask(frame: np.ndarray) -> np.ndarray:
	"""Return the established clean-DMC blue-background predicate."""
	if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
		raise ValueError(
			"Foreground erosion expects one uint8 HWC RGB frame, got "
			f"shape={frame.shape}, dtype={frame.dtype}."
		)
	return (frame[..., 2] > frame[..., 1]) & (frame[..., 2] > frame[..., 0])


def _binary_erode_square(mask: np.ndarray, radius: int) -> np.ndarray:
	"""Erode a 2-D bool mask with a square radius using NumPy only."""
	if mask.ndim != 2 or mask.dtype != np.bool_:
		raise ValueError("Binary erosion requires a two-dimensional bool mask.")
	radius = int(radius)
	if radius < 0:
		raise ValueError(f"Binary erosion radius must be non-negative, got {radius}.")
	if radius == 0:
		return mask.copy()
	result = np.ones_like(mask, dtype=np.bool_)
	padded = np.pad(mask, radius, mode="constant", constant_values=False)
	height, width = mask.shape
	for dy in range(2 * radius + 1):
		for dx in range(2 * radius + 1):
			result &= padded[dy:dy + height, dx:dx + width]
	return result


def _erode_clean_rgb_frame(frame: np.ndarray, radius: int) -> np.ndarray:
	"""Erode one clean-DMC HWC frame at an explicitly scaled radius."""
	frame = np.asarray(frame)
	background = _blue_background_mask(frame)
	if not background.any():
		raise ValueError(
			"Clean RGB frame contains no blue background pixels; foreground "
			"erosion must run before video-background composition."
		)
	foreground = ~background
	eroded = _binary_erode_square(foreground, radius)
	removed = foreground & ~eroded
	fill = np.median(frame[background], axis=0).astype(np.uint8)
	stressed = np.array(frame, dtype=np.uint8, order="C", copy=True)
	stressed[removed] = fill
	return stressed


def erode_clean_rgb_stack(observation, erosion_pixels: int):
	"""Return an eroded copy of one CHW RGB stack.

	The return type matches the input type (``torch.Tensor`` or
	``numpy.ndarray``).  The input is never mutated.
	"""
	radius = int(erosion_pixels)
	if radius != erosion_pixels or radius not in _ALLOWED_EROSION_PIXELS:
		raise ValueError(
			f"erosion_pixels must be one of {_ALLOWED_EROSION_PIXELS}, "
			f"got {erosion_pixels!r}."
		)
	is_tensor = torch.is_tensor(observation)
	if is_tensor:
		if observation.device.type != "cpu":
			raise ValueError("Foreground erosion expects the Pixels CPU tensor.")
		array = observation.detach().numpy()
	elif isinstance(observation, np.ndarray):
		array = observation
	else:
		raise TypeError(
			"Foreground erosion observation must be a torch.Tensor or numpy.ndarray."
		)
	if (
		array.ndim != 3
		or array.shape[0] == 0
		or array.shape[0] % 3
		or array.dtype != np.uint8
	):
		raise ValueError(
			"Foreground erosion expects uint8 CHW stacked RGB with channels "
			f"divisible by three, got shape={array.shape}, dtype={array.dtype}."
		)
	output = np.array(array, dtype=np.uint8, order="C", copy=True)
	if radius:
		for start in range(0, output.shape[0], 3):
			frame = np.moveaxis(array[start:start + 3], 0, -1)
			stressed = _erode_clean_rgb_frame(frame, radius)
			output[start:start + 3] = np.moveaxis(stressed, -1, 0)
	if is_tensor:
		return torch.from_numpy(output)
	return output


class ForegroundErosionWrapper(gym.Wrapper):
	"""Apply 0/1/2-pixel foreground erosion to every clean RGB observation."""

	def __init__(self, env, erosion_pixels: int):
		super().__init__(env)
		radius = int(erosion_pixels)
		if radius != erosion_pixels or radius not in _ALLOWED_EROSION_PIXELS:
			raise ValueError(
				f"erosion_pixels must be one of {_ALLOWED_EROSION_PIXELS}, "
				f"got {erosion_pixels!r}."
			)
		shape = tuple(env.observation_space.shape)
		if (
			len(shape) != 3
			or shape[0] == 0
			or shape[0] % 3
			or np.dtype(env.observation_space.dtype) != np.dtype(np.uint8)
		):
			raise ValueError(
				"ForegroundErosionWrapper requires a uint8 CHW RGB stack, got "
				f"shape={shape}, dtype={env.observation_space.dtype}."
			)
		self._erosion_pixels = radius
		self.observation_space = env.observation_space

	@property
	def erosion_pixels(self) -> int:
		"""Frozen radius exposed for evaluator provenance and fail-closed checks."""
		return self._erosion_pixels

	def _stress(self, observation):
		return erode_clean_rgb_stack(observation, self.erosion_pixels)

	def cutie_same_state_rgb(self, *, height, width):
		"""Apply the same foreground stress to an extra same-state RGB render.

		The configured radius is expressed in policy-image pixels.  Scale it with
		the requested render so a one-pixel 64x64 intervention remains the same
		fraction of the image at 128x128 or 256x256.
		"""
		render = getattr(self.env, 'cutie_same_state_rgb', None)
		if not callable(render):
			raise RuntimeError(
				'Foreground stress cannot obtain a same-state RGB render from its child.'
			)
		frame = render(height=height, width=width)
		base_height, base_width = self.observation_space.shape[-2:]
		if height * base_width != width * base_height:
			raise ValueError('Cutie high-resolution render must preserve aspect ratio.')
		scale = float(height) / float(base_height)
		radius = int(round(self.erosion_pixels * scale))
		return _erode_clean_rgb_frame(frame, radius)

	def reset(self, **kwargs):
		result = self.env.reset(**kwargs)
		# The TD-MPC2 wrappers use the legacy observation-only reset API, while
		# accepting Gymnasium's (observation, info) form keeps this wrapper local.
		if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
			return self._stress(result[0]), result[1]
		return self._stress(result)

	def step(self, action):
		result = self.env.step(action)
		if not isinstance(result, tuple) or len(result) not in (4, 5):
			raise TypeError("Wrapped environment step must return a 4- or 5-tuple.")
		return (self._stress(result[0]), *result[1:])


# Descriptive alias for callers that treat erosion as one visual stress mode.
ForegroundStressWrapper = ForegroundErosionWrapper


def _self_contract() -> None:
	"""Small deterministic contract that does not require dm_control."""
	frame = np.empty((9, 9, 3), dtype=np.uint8)
	frame[...] = np.array([20, 40, 120], dtype=np.uint8)
	frame[2:7, 2:7] = np.array([180, 50, 30], dtype=np.uint8)
	stack = np.concatenate(
		[np.moveaxis(frame + offset, -1, 0) for offset in (0, 1, 2)], axis=0
	).astype(np.uint8)
	before = stack.copy()
	zero = erode_clean_rgb_stack(stack, 0)
	one = erode_clean_rgb_stack(torch.from_numpy(stack), 1)
	two = erode_clean_rgb_stack(stack, 2)
	assert isinstance(zero, np.ndarray) and isinstance(one, torch.Tensor)
	assert zero.shape == one.shape == two.shape == stack.shape
	assert zero.dtype == two.dtype == np.uint8 and one.dtype == torch.uint8
	assert np.array_equal(stack, before), "The source observation was mutated."
	assert np.array_equal(zero, stack), "Radius zero must be bitwise identity."
	counts = []
	for value in (zero, one.numpy(), two):
		latest = np.moveaxis(value[-3:], 0, -1)
		counts.append(int((~_blue_background_mask(latest)).sum()))
	assert counts == [25, 9, 1], counts
	print(
		"FOREGROUND_STRESS_CONTRACT_OK",
		{"shape": tuple(stack.shape), "foreground_counts": counts},
	)


if __name__ == "__main__":
	_self_contract()

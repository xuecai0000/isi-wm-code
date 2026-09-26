"""Dynamic color-video backgrounds with leakage-safe source splits.

The manifests in :mod:`envs.background_manifests` contain filenames only. The
large video files stay in the existing external HRSSM data directory and are
resolved lazily when this wrapper is explicitly enabled.
"""

from collections import OrderedDict, deque
import hashlib
import json
from pathlib import Path
import re

import gymnasium as gym
import numpy as np
from PIL import Image
import torch


_SPLITS = ('train', 'validation', 'test', 'support')
_EXPECTED_SOURCE_INDICES_BY_SCHEMA = {
	# Historical Color-multi contract. Never reinterpret these manifests.
	1: {
		'train': range(0, 70),
		'validation': range(70, 80),
		'test': range(80, 85),
		'support': range(85, 90),
	},
	# Common-background comparison contract. Legacy auxiliary videos 70--79
	# become training assets, while a new disjoint 90--99 pool is the only test
	# domain.  This profile lives in a separate manifest directory.
	2: {
		'train': range(0, 80),
		'validation': range(80, 85),
		'test': range(90, 100),
		'support': range(85, 90),
	},
}
_VIDEO_NAME = re.compile(r'video(\d+)\.mp4')


def _config_value(cfg, name, default=None):
	try:
		return cfg.get(name, default)
	except AttributeError:
		return getattr(cfg, name, default)


class ColorMultiSplitSelector:
	"""Resolve one explicit, mutually exclusive color-video source split."""

	def __init__(self, video_root, manifest_dir=None):
		if not video_root:
			raise ValueError(
				'video_background_root must reference the existing video_hard directory.'
			)
		self._video_root = Path(str(video_root)).expanduser().resolve()
		if not self._video_root.is_dir():
			raise FileNotFoundError(
				f'Color-multi video root is not a directory: {self._video_root}'
			)
		self._manifest_dir = (
			Path(str(manifest_dir)).expanduser().resolve()
			if manifest_dir
			else Path(__file__).resolve().parents[1] / 'background_manifests'
		)
		self._sources = self._load_manifests()

	@property
	def video_root(self):
		return self._video_root

	@property
	def manifest_dir(self):
		return self._manifest_dir

	@property
	def splits(self):
		return _SPLITS

	@property
	def manifest_sha256s(self):
		return dict(self._manifest_sha256s)

	@property
	def combined_manifest_sha256(self):
		return self._combined_manifest_sha256

	@property
	def manifest_schema_version(self):
		return self._manifest_schema_version

	def manifest_sha256(self, split):
		self.source_names(split)
		return self._manifest_sha256s[str(split)]

	def source_names(self, split):
		split = str(split)
		if split not in self._sources:
			raise ValueError(
				f'Unknown video_background_split {split!r}; expected one of {_SPLITS}.'
			)
		return self._sources[split]

	def resolve(self, split):
		"""Return existing external paths for exactly one manifest split."""
		paths = []
		for name in self.source_names(split):
			path = (self._video_root / name).resolve()
			if path.parent != self._video_root:
				raise ValueError(f'Video manifest path escapes its root: {name!r}.')
			if not path.is_file():
				raise FileNotFoundError(
					f'Manifest source {name!r} is missing under {self._video_root}.'
				)
			paths.append(path)
		return tuple(paths)

	def _load_manifests(self):
		loaded = {}
		owners = {}
		self._manifest_sha256s = {}
		schema_version = None
		for expected_split in _SPLITS:
			path = self._manifest_dir / f'color_multi_{expected_split}.json'
			if not path.is_file():
				raise FileNotFoundError(f'Color-multi split manifest is missing: {path}')
			raw = path.read_bytes()
			payload = json.loads(raw.decode('utf-8'))
			self._manifest_sha256s[expected_split] = hashlib.sha256(raw).hexdigest()
			if set(payload) != {'schema_version', 'name', 'sources'}:
				raise ValueError(
					f'{path} must contain exactly schema_version, name, and sources.'
				)
			if payload['schema_version'] not in _EXPECTED_SOURCE_INDICES_BY_SCHEMA:
				raise ValueError(
					f'{path} has unsupported schema_version {payload["schema_version"]!r}.'
				)
			if schema_version is None:
				schema_version = int(payload['schema_version'])
			elif int(payload['schema_version']) != schema_version:
				raise ValueError('Color-multi manifests mix incompatible schema versions.')
			if payload['name'] != expected_split:
				raise ValueError(
					f'{path} declares split {payload["name"]!r}, expected {expected_split!r}.'
				)
			sources = payload['sources']
			if not isinstance(sources, list) or not sources:
				raise ValueError(f'{path} sources must be a non-empty list.')
			if any(not isinstance(name, str) or Path(name).name != name for name in sources):
				raise ValueError(f'{path} sources must be plain relative filenames.')
			if len(sources) != len(set(sources)):
				raise ValueError(f'{path} contains duplicate sources.')

			indices = []
			for name in sources:
				match = _VIDEO_NAME.fullmatch(name)
				if match is None:
					raise ValueError(
						f'{path} source {name!r} does not match video<index>.mp4.'
					)
				index = int(match.group(1))
				if name != f'video{index}.mp4':
					raise ValueError(
						f'{path} source {name!r} is not a canonical video filename.'
					)
				if name in owners:
					raise ValueError(
						f'Color-multi source {name!r} overlaps {owners[name]!r} and '
						f'{expected_split!r}.'
					)
				owners[name] = expected_split
				indices.append(index)

			expected_indices = tuple(
				_EXPECTED_SOURCE_INDICES_BY_SCHEMA[schema_version][expected_split]
			)
			if tuple(indices) != expected_indices:
				raise ValueError(
					f'{path} must list exactly indices {expected_indices[0]}-'
					f'{expected_indices[-1]} in numeric order; got {tuple(indices)}.'
				)
			loaded[expected_split] = tuple(sources)

		combined = ''.join(
			f'{split}:{self._manifest_sha256s[split]}\n' for split in _SPLITS
		).encode('ascii')
		self._combined_manifest_sha256 = hashlib.sha256(combined).hexdigest()
		self._manifest_schema_version = int(schema_version)
		return loaded


class VideoBackgroundCompositor:
	"""Replace DMC's blue render background with one sequential video stream."""

	def __init__(
		self,
		selector,
		split,
		size=(64, 64),
		strength=1.0,
		total_frames=1000,
		source_cache_size=8,
		seed=0,
	):
		self._selector = selector
		self._split = str(split)
		self._sources = selector.resolve(self._split)
		self._source_names = selector.source_names(self._split)
		self._size = tuple(int(value) for value in size)
		if len(self._size) != 2 or min(self._size) <= 0:
			raise ValueError(f'Video background size must be positive HxW, got {size}.')
		self._strength = float(strength)
		if not 0. <= self._strength <= 1.:
			raise ValueError('video_background_strength must be in [0, 1].')
		self._total_frames = None if total_frames is None else int(total_frames)
		if self._total_frames is not None and self._total_frames <= 0:
			raise ValueError('video_background_total_frames must be positive or null.')
		self._source_cache_size = int(source_cache_size)
		if self._source_cache_size < 0:
			raise ValueError('video_background_source_cache_size must be non-negative.')
		self._random = np.random.RandomState(int(seed))
		self._source_cache = OrderedDict()
		self._active_source = None
		self._frames = None
		self._start = 0
		self._frame = 0
		self._last_frame_index = None

	@property
	def active_source(self):
		return self._active_source

	@property
	def active_split(self):
		return self._split

	@property
	def source_names(self):
		return self._source_names

	@property
	def manifest_sha256(self):
		return self._selector.manifest_sha256(self._split)

	@property
	def combined_manifest_sha256(self):
		return self._selector.combined_manifest_sha256

	@property
	def frame_index(self):
		"""Index of the most recently used frame within the active source."""
		return self._last_frame_index

	def reset(self):
		self._active_source = self._sources[
			self._random.randint(0, len(self._sources))
		]
		self._frames = self._load_source_frames(self._active_source)
		self._start = int(self._random.randint(0, len(self._frames)))
		self._frame = 0
		self._last_frame_index = None

	def apply(self, clean_image):
		if self._frames is None:
			self.reset()
		clean = np.asarray(clean_image)
		expected_shape = (self._size[0], self._size[1], 3)
		if clean.shape != expected_shape or clean.dtype != np.uint8:
			raise ValueError(
				f'Video compositor expects uint8 HWC {expected_shape}, '
				f'got shape={clean.shape}, dtype={clean.dtype}.'
			)
		index = (self._start + self._frame) % len(self._frames)
		self._frame += 1
		self._last_frame_index = int(index)
		return self._replace_background(clean, self._frames[index])

	def apply_current(self, clean_image):
		"""Compose a second-resolution render with the current video frame.

		Unlike :meth:`apply`, this method never advances the sequential background
		clock.  It is used only for Cutie's optional same-state RGB render after the
		ordinary policy frame has selected and consumed the current video index.
		"""
		if self._frames is None or self._last_frame_index is None:
			raise RuntimeError(
				'No current background frame exists; compose the policy frame first.'
			)
		clean = np.asarray(clean_image)
		if clean.ndim != 3 or clean.shape[-1] != 3 or clean.dtype != np.uint8:
			raise ValueError(
				'Video compositor expects one uint8 HWC RGB frame, got '
				f'{clean.shape} {clean.dtype}.'
			)
		background = self._frames[self._last_frame_index]
		if background.shape != clean.shape:
			resampling = getattr(Image, 'Resampling', Image).BILINEAR
			background = np.asarray(
				Image.fromarray(background).resize(
					(clean.shape[1], clean.shape[0]), resampling
				),
				dtype=np.uint8,
			)
		return self._replace_background(clean, background)

	@staticmethod
	def _background_mask(image):
		# Match the established R2/HRSSM compositor: default DMC backgrounds are blue.
		return np.logical_and(
			image[..., 2] > image[..., 1],
			image[..., 2] > image[..., 0],
		)

	def _replace_background(self, clean, background):
		output = clean.copy()
		mask = self._background_mask(clean)
		if self._strength == 1.:
			output[mask] = background[mask]
		else:
			mixed = (
				self._strength * background.astype(np.float32)
				+ (1. - self._strength) * clean.astype(np.float32)
			).astype(np.uint8)
			output[mask] = mixed[mask]
		return output

	def _load_source_frames(self, source):
		if source in self._source_cache:
			frames = self._source_cache.pop(source)
			self._source_cache[source] = frames
			return frames

		frames = self._read_source(source)
		if not frames:
			raise ValueError(f'No readable frames found in color-multi source {source}.')
		frames = np.stack(frames, axis=0).astype(np.uint8, copy=False)
		if self._source_cache_size > 0:
			self._source_cache[source] = frames
			while len(self._source_cache) > self._source_cache_size:
				self._source_cache.popitem(last=False)
		return frames

	def _read_source(self, source):
		errors = []
		for reader in (
			self._read_video_cv2,
			self._read_video_imageio_ffmpeg,
			self._read_video_imageio,
			self._read_video_moviepy,
		):
			try:
				frames = reader(source)
			except (ImportError, OSError, RuntimeError, ValueError) as exc:
				errors.append(exc)
				continue
			if frames:
				return frames
			errors.append(ValueError(f'{reader.__name__} decoded zero frames.'))
		details = '; '.join(str(error) for error in errors)
		raise ValueError(
			f'Could not read color-multi video {source}; tried cv2, direct '
			f'imageio-ffmpeg, imageio, and moviepy. {details}'
		)

	def _read_video_cv2(self, source):
		import cv2

		capture = cv2.VideoCapture(str(source))
		if not capture.isOpened():
			raise ValueError(f'Could not open video file: {source}')
		frames = []
		try:
			ok, frame = capture.read()
			while ok:
				frames.append(self._prepare_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
				if self._limit_reached(frames):
					break
				ok, frame = capture.read()
		finally:
			capture.release()
		return frames

	def _read_video_imageio(self, source):
		import imageio.v3 as iio

		frames = []
		for frame in iio.imiter(source):
			frames.append(self._prepare_frame(frame))
			if self._limit_reached(frames):
				break
		return frames

	def _read_video_imageio_ffmpeg(self, source):
		"""Decode through the already-installed imageio-ffmpeg backend directly.

		Some imageio releases do not auto-register their ffmpeg plugin even when
		``imageio-ffmpeg`` is installed. Calling its byte-stream reader directly
		avoids optional OpenCV, PyAV, and MoviePy decoder dependencies. The pinned
		imageio-ffmpeg release still needs ``pkg_resources`` from setuptools.
		"""
		import imageio_ffmpeg

		reader = imageio_ffmpeg.read_frames(
			str(source),
			pix_fmt='rgb24',
			bits_per_pixel=24,
		)
		frames = []
		try:
			try:
				metadata = next(reader)
			except StopIteration as exc:
				raise ValueError(f'No video metadata returned for {source}.') from exc
			size = metadata.get('size')
			if not isinstance(size, (tuple, list)) or len(size) != 2:
				raise ValueError(
					f'imageio-ffmpeg returned invalid frame size {size!r} for {source}.'
				)
			width, height = (int(size[0]), int(size[1]))
			if width <= 0 or height <= 0:
				raise ValueError(
					f'imageio-ffmpeg returned non-positive frame size {size!r} for {source}.'
				)
			expected_bytes = width * height * 3
			for raw_frame in reader:
				if len(raw_frame) != expected_bytes:
					raise ValueError(
						f'Unexpected decoded frame length for {source}: '
						f'{len(raw_frame)} != {expected_bytes}.'
					)
				frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
					height, width, 3
				)
				frames.append(self._prepare_frame(frame))
				if self._limit_reached(frames):
					break
		finally:
			reader.close()
		return frames

	def _read_video_moviepy(self, source):
		from moviepy.editor import VideoFileClip

		frames = []
		clip = VideoFileClip(str(source), audio=False)
		try:
			for frame in clip.iter_frames(dtype='uint8'):
				frames.append(self._prepare_frame(frame))
				if self._limit_reached(frames):
					break
		finally:
			clip.close()
		return frames

	def _limit_reached(self, frames):
		return self._total_frames is not None and len(frames) >= self._total_frames

	def _prepare_frame(self, frame):
		resampling = getattr(Image, 'Resampling', Image).BILINEAR
		image = Image.fromarray(np.asarray(frame).astype(np.uint8, copy=False))
		image = image.convert('RGB').resize(
			(self._size[1], self._size[0]),
			resampling,
		)
		return np.asarray(image, dtype=np.uint8)


class ColorMultiVideoBackgroundWrapper(gym.Wrapper):
	"""Compose only the newest RGB frame and retain a causal three-frame stack."""

	def __init__(self, env, cfg):
		super().__init__(env)
		shape = tuple(env.observation_space.shape)
		if len(shape) != 3 or shape[0] % 3 or shape[1] != shape[2]:
			raise ValueError(
				'Color-multi wrapper expects stacked CHW RGB observations, '
				f'got {shape}.'
			)
		self._num_frames = shape[0] // 3
		self._size = (shape[1], shape[2])
		selector = ColorMultiSplitSelector(
			_config_value(cfg, 'video_background_root'),
			_config_value(cfg, 'video_background_manifest_dir'),
		)
		self._compositor = VideoBackgroundCompositor(
			selector=selector,
			split=_config_value(cfg, 'video_background_split', 'train'),
			size=self._size,
			strength=_config_value(cfg, 'video_background_strength', 1.0),
			total_frames=_config_value(cfg, 'video_background_total_frames', 1000),
			source_cache_size=_config_value(
				cfg, 'video_background_source_cache_size', 8
			),
			# Training keeps the historical behavior because the optional
			# background seed falls back to ``cfg.seed``.  Standalone causal
			# audits can set it explicitly so background selection is independent
			# of the simulator/physics seed.
			seed=_config_value(
				cfg,
				'video_background_seed',
				_config_value(cfg, 'seed', 0),
			),
		)
		self._frames = deque(maxlen=self._num_frames)
		self._last_frame = None

	@property
	def active_source(self):
		return self._compositor.active_source

	@property
	def active_split(self):
		return self._compositor.active_split

	@property
	def source_names(self):
		return self._compositor.source_names

	@property
	def manifest_sha256(self):
		return self._compositor.manifest_sha256

	@property
	def combined_manifest_sha256(self):
		return self._compositor.combined_manifest_sha256

	@property
	def frame_index(self):
		return self._compositor.frame_index

	@property
	def compositor(self):
		return self._compositor

	@staticmethod
	def _latest_clean_frame(observation):
		observation = torch.as_tensor(observation)
		if observation.ndim != 3 or observation.shape[0] < 3:
			raise ValueError(
				f'Expected stacked CHW RGB observation, got {tuple(observation.shape)}.'
			)
		return (
			observation[-3:]
			.detach()
			.cpu()
			.permute(1, 2, 0)
			.contiguous()
			.numpy()
		)

	def _compose_latest(self, observation):
		frame = self._compositor.apply(self._latest_clean_frame(observation))
		self._last_frame = frame
		return torch.from_numpy(frame.copy()).permute(2, 0, 1).contiguous()

	def cutie_same_state_rgb(self, *, height, width):
		"""Replay the selected background frame on a same-state native render."""
		render = getattr(self.env, 'cutie_same_state_rgb', None)
		if not callable(render):
			raise RuntimeError(
				'Video background cannot obtain a same-state RGB render from its child.'
			)
		clean = render(height=height, width=width)
		return np.array(
			self._compositor.apply_current(clean),
			dtype=np.uint8,
			order='C',
			copy=True,
		)

	def reset(self):
		clean_observation = self.env.reset()
		self._compositor.reset()
		frame = self._compose_latest(clean_observation)
		self._frames.clear()
		for _ in range(self._num_frames):
			self._frames.append(frame.clone())
		return torch.cat(tuple(self._frames), dim=0)

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		frame = self._compose_latest(observation)
		if not self._frames:
			for _ in range(self._num_frames - 1):
				self._frames.append(frame.clone())
		self._frames.append(frame)
		return torch.cat(tuple(self._frames), dim=0), reward, done, info

	def render(self, width=384, height=384, camera_id=None):
		"""Return the cached composed frame without advancing the video stream."""
		if self._last_frame is None:
			return self.env.render(width=width, height=height, camera_id=camera_id)
		frame = self._last_frame
		if (height, width) != frame.shape[:2]:
			resampling = getattr(Image, 'Resampling', Image).BILINEAR
			frame = np.asarray(
				Image.fromarray(frame).resize((width, height), resampling),
				dtype=np.uint8,
			)
		return frame.copy()

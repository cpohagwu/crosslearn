from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from crosslearn.envs import AtariPreprocessor
from crosslearn.extractors import NatureCNNExtractor


pytest.importorskip("cv2")


class _FakeALE:
    def __init__(self, env: "_FakeAtariEnv") -> None:
        self.env = env

    def lives(self) -> int:
        return self.env._lives

    def getScreenGrayscale(self, out: np.ndarray) -> None:
        out[...] = self.env._gray_frame

    def getScreenRGB(self, out: np.ndarray) -> None:
        out[...] = self.env._rgb_frame


class _FakeAtariEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=np.uint8,
        )
        self.action_space = Discrete(4)
        self._frameskip = 1
        self._lives = 3
        self._step_count = 0
        self._gray_frame = np.zeros((210, 160), dtype=np.uint8)
        self._rgb_frame = np.zeros((210, 160, 3), dtype=np.uint8)
        self.ale = _FakeALE(self)

    def get_action_meanings(self) -> list[str]:
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def _set_frame(self, value: int) -> None:
        pixel_value = np.uint8(value % 256)
        self._gray_frame.fill(pixel_value)
        self._rgb_frame.fill(pixel_value)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._lives = 3
        self._set_frame(0)
        return self._rgb_frame.copy(), {}

    def step(self, action: int):
        self._step_count += 1
        self._set_frame(self._step_count)
        reward = 1.0
        terminated = self._step_count >= 100
        truncated = False
        return self._rgb_frame.copy(), reward, terminated, truncated, {}


def test_atari_preprocessor_stacks_channel_first_frames() -> None:
    env = AtariPreprocessor(_FakeAtariEnv(), noop_max=0, stack_size=4, frame_skip=1)

    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8
    assert obs[:, 0, 0].tolist() == [0, 0, 0, 0]

    next_obs, _, _, _, _ = env.step(0)
    assert next_obs.shape == (4, 84, 84)
    assert next_obs.dtype == np.uint8
    assert next_obs[:, 0, 0].tolist() == [0, 0, 0, 1]


def test_atari_preprocessor_defaults_preserve_raw_rewards() -> None:
    env = AtariPreprocessor(_FakeAtariEnv())

    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8

    next_obs, reward, terminated, truncated, _ = env.step(0)
    assert next_obs.shape == (4, 84, 84)
    assert next_obs.dtype == np.uint8
    assert reward == pytest.approx(1.0)
    assert not terminated
    assert not truncated


def test_atari_preprocessor_frame_skip_sums_rewards() -> None:
    env = AtariPreprocessor(_FakeAtariEnv(), noop_max=0, frame_skip=4)

    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84)

    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == pytest.approx(4.0)
    assert not terminated
    assert not truncated


def test_nature_cnn_accepts_preprocessed_atari_observations() -> None:
    env = AtariPreprocessor(_FakeAtariEnv(), noop_max=0, stack_size=4, frame_skip=1)
    obs, _ = env.reset(seed=0)

    extractor = NatureCNNExtractor(env.observation_space, features_dim=512)
    batch = torch.from_numpy(np.stack([obs, obs], axis=0))

    features = extractor(batch)

    assert features.shape == (2, 512)


def test_nature_cnn_handles_raw_and_normalized_images_identically() -> None:
    env = AtariPreprocessor(_FakeAtariEnv(), noop_max=0, stack_size=4, frame_skip=1)
    obs, _ = env.reset(seed=0)

    extractor = NatureCNNExtractor(env.observation_space, features_dim=512)
    raw_batch = torch.from_numpy(np.stack([obs, obs], axis=0))
    normalized_batch = raw_batch.float() / 255.0

    raw_features = extractor(raw_batch)
    normalized_features = extractor(normalized_batch)

    torch.testing.assert_close(raw_features, normalized_features, rtol=0.0, atol=1e-6)

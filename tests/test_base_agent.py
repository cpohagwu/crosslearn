from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import crosslearn.agents.base as base_module
from crosslearn._devices import resolve_device, resolve_device_map
from crosslearn.agents.base import BaseAgent


class _ConstantEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: int):
        return np.zeros(4, dtype=np.float32), 0.0, True, False, {}


class _DummyAgent(BaseAgent):
    def _build_policy(self) -> nn.Module:
        return nn.Linear(self.observation_space.shape[0], self.action_space.n).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def _collect_rollout(self) -> dict:
        return {"n_steps": 0, "n_episodes": 0, "episode_rewards": [], "mean_episode_reward": 0.0}

    def _update_policy(self, rollout: dict) -> dict:
        return {}

    def _predict_action(self, obs_t: torch.Tensor, deterministic: bool) -> np.ndarray:
        return np.zeros(obs_t.shape[0], dtype=np.int64)

    def __repr__(self) -> str:
        return "_DummyAgent()"


def test_resolve_device_honors_explicit_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_honors_explicit_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda").type == "cuda"
    assert resolve_device_map("cuda") == "cuda"


def test_resolve_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    assert resolve_device_map("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    assert resolve_device_map("auto") == "cpu"


def test_base_agent_honors_explicit_cpu_when_cuda_is_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    agent = _DummyAgent(
        lambda: _ConstantEnv(),
        n_envs=1,
        device="cpu",
        verbose=0,
    )

    assert agent.device.type == "cpu"


def test_base_agent_passes_use_async_env_to_make_vec_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeVecEnv:
        num_envs = 2
        single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )
        single_action_space = gym.spaces.Discrete(2)

    def _fake_make_vec_env(env_input, n_envs=1, use_async=False):
        captured["env_input"] = env_input
        captured["n_envs"] = n_envs
        captured["use_async"] = use_async
        return _FakeVecEnv()

    monkeypatch.setattr(base_module, "make_vec_env", _fake_make_vec_env)

    agent = _DummyAgent(
        "CartPole-v1",
        n_envs=2,
        use_async_env=True,
        verbose=0,
    )

    assert agent.use_async_env is True
    assert captured["env_input"] == "CartPole-v1"
    assert captured["n_envs"] == 2
    assert captured["use_async"] is True

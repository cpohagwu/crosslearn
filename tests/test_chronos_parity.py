"""
They check:

same extractor features, logits, values, and greedy actions on a fixed observation trace,
same deterministic episode action sequence through the actual REINFORCE.predict(..., deterministic=True) path,
and that a deliberately wrong Chronos variant is detected.
"""

from __future__ import annotations

from typing import Type

import gymnasium as gym
import numpy as np
import pytest
import torch

from crosslearn import REINFORCE
from crosslearn.extractors.chronos import ChronosExtractor
from crosslearn.policies.actor_critic import ActorCriticPolicy


FEATURE_NAMES = ["Open", "High", "Low", "Close", "Volume"]
SELECTED_COLUMNS = ["Close", "Volume"]
LOOKBACK = 4
N_FEATURES = len(FEATURE_NAMES)


class _FeaturePermutingChronosExtractor(ChronosExtractor):
    """Deliberately wrong extractor used to prove the parity harness catches drift."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = super().forward(observations)
        return torch.flip(features, dims=[-1])


class _ReplayTradingEnv(gym.Env):
    metadata = {}

    def __init__(self, observations: np.ndarray) -> None:
        self._observations = observations.astype(np.float32, copy=True)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._observations.shape[1:],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self._index = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._index = 0
        return self._observations[self._index].copy(), {}

    def step(self, action: int):
        self._index += 1
        terminated = self._index >= len(self._observations)
        next_index = min(self._index, len(self._observations) - 1)
        reward = float(action)
        return (
            self._observations[next_index].copy(),
            reward,
            terminated,
            False,
            {"action": int(action), "index": next_index},
        )


def _make_observation_trace(
    *,
    n_steps: int = 6,
    lookback: int = LOOKBACK,
    n_features: int = N_FEATURES,
) -> torch.Tensor:
    values = torch.arange(n_steps * lookback * n_features, dtype=torch.float32)
    return values.reshape(n_steps, lookback, n_features) / 10.0


def _make_policy(
    *,
    observation_space: gym.Space,
    action_space: gym.spaces.Discrete,
    features_extractor_class: Type[ChronosExtractor],
) -> ActorCriticPolicy:
    return ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[],
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs={
            "feature_names": FEATURE_NAMES,
            "selected_columns": SELECTED_COLUMNS,
        },
    )


def _configure_policy_for_action_sensitivity(policy: ActorCriticPolicy) -> None:
    with torch.no_grad():
        policy.actor_head.weight.zero_()
        policy.actor_head.bias.zero_()
        policy.actor_head.weight[0, 0] = 1.0
        policy.actor_head.weight[1, -1] = 1.0
        policy.critic_head.weight.zero_()
        policy.critic_head.bias.zero_()


def _capture_policy_trace(
    policy: ActorCriticPolicy,
    observations: torch.Tensor,
) -> dict[str, torch.Tensor]:
    policy.eval()
    with torch.no_grad():
        features = policy.features_extractor(observations)
        logits, values = policy(observations)
        greedy_actions = policy.predict_actions(observations, deterministic=True)
    return {
        "features": features.detach().cpu(),
        "logits": logits.detach().cpu(),
        "values": values.detach().cpu(),
        "greedy_actions": greedy_actions.detach().cpu(),
    }


def _assert_policy_parity(
    *,
    reference_class: Type[ChronosExtractor],
    candidate_class: Type[ChronosExtractor],
    observations: torch.Tensor,
) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(LOOKBACK, N_FEATURES),
        dtype=np.float32,
    )
    action_space = gym.spaces.Discrete(2)

    reference_policy = _make_policy(
        observation_space=observation_space,
        action_space=action_space,
        features_extractor_class=reference_class,
    )
    candidate_policy = _make_policy(
        observation_space=observation_space,
        action_space=action_space,
        features_extractor_class=candidate_class,
    )

    candidate_policy.load_state_dict(reference_policy.state_dict())
    _configure_policy_for_action_sensitivity(reference_policy)
    _configure_policy_for_action_sensitivity(candidate_policy)

    reference_trace = _capture_policy_trace(reference_policy, observations)
    candidate_trace = _capture_policy_trace(candidate_policy, observations)

    torch.testing.assert_close(candidate_trace["features"], reference_trace["features"])
    torch.testing.assert_close(candidate_trace["logits"], reference_trace["logits"])
    torch.testing.assert_close(candidate_trace["values"], reference_trace["values"])
    torch.testing.assert_close(
        candidate_trace["greedy_actions"], reference_trace["greedy_actions"]
    )


def _make_agent(
    *,
    env: gym.Env,
    features_extractor_class: Type[ChronosExtractor],
) -> REINFORCE:
    agent = REINFORCE(
        env,
        n_envs=1,
        n_steps=4,
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs={
            "feature_names": FEATURE_NAMES,
            "selected_columns": SELECTED_COLUMNS,
        },
        policy_kwargs={"net_arch": []},
        verbose=0,
        seed=7,
    )
    _configure_policy_for_action_sensitivity(agent.policy)
    return agent


def _run_deterministic_episode(agent: REINFORCE, env: gym.Env) -> list[int]:
    obs, _ = env.reset(seed=0)
    terminated = False
    truncated = False
    actions: list[int] = []

    while not (terminated or truncated):
        action = int(agent.predict(obs, deterministic=True))
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)

    return actions


def test_chronos_policy_parity_harness_matches_reference_extractor(fake_chronos) -> None:
    trace = _make_observation_trace()
    _assert_policy_parity(
        reference_class=ChronosExtractor,
        candidate_class=ChronosExtractor,
        observations=trace,
    )


def test_chronos_policy_parity_harness_detects_feature_and_action_drift(
    fake_chronos,
) -> None:
    trace = _make_observation_trace()

    with pytest.raises(AssertionError):
        _assert_policy_parity(
            reference_class=ChronosExtractor,
            candidate_class=_FeaturePermutingChronosExtractor,
            observations=trace,
        )


def test_chronos_render_parity_harness_matches_reference_agent(fake_chronos) -> None:
    observations = _make_observation_trace().numpy()

    reference_agent = _make_agent(
        env=_ReplayTradingEnv(observations),
        features_extractor_class=ChronosExtractor,
    )
    candidate_agent = _make_agent(
        env=_ReplayTradingEnv(observations),
        features_extractor_class=ChronosExtractor,
    )
    candidate_agent.policy.load_state_dict(reference_agent.policy.state_dict())

    reference_actions = _run_deterministic_episode(
        reference_agent,
        _ReplayTradingEnv(observations),
    )
    candidate_actions = _run_deterministic_episode(
        candidate_agent,
        _ReplayTradingEnv(observations),
    )

    assert candidate_actions == reference_actions


def test_chronos_render_parity_harness_detects_action_drift(fake_chronos) -> None:
    observations = _make_observation_trace().numpy()

    reference_agent = _make_agent(
        env=_ReplayTradingEnv(observations),
        features_extractor_class=ChronosExtractor,
    )
    candidate_agent = _make_agent(
        env=_ReplayTradingEnv(observations),
        features_extractor_class=_FeaturePermutingChronosExtractor,
    )
    candidate_agent.policy.load_state_dict(reference_agent.policy.state_dict())

    reference_actions = _run_deterministic_episode(
        reference_agent,
        _ReplayTradingEnv(observations),
    )
    candidate_actions = _run_deterministic_episode(
        candidate_agent,
        _ReplayTradingEnv(observations),
    )

    assert candidate_actions != reference_actions

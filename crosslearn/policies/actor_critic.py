from __future__ import annotations

from typing import List, Optional, Tuple, Type

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

from crosslearn.extractors.base import BaseFeaturesExtractor
from crosslearn.extractors.flatten import FlattenExtractor


def _build_mlp(
    input_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module],
) -> Tuple[nn.Sequential, int]:
    """Build an MLP. Returns (net, output_dim)."""
    if not net_arch:
        return nn.Identity(), input_dim  # type: ignore[return-value]
    layers: List[nn.Module] = []
    last = input_dim
    for hidden in net_arch:
        layers += [nn.Linear(last, hidden), activation_fn()]
        last = hidden
    return nn.Sequential(*layers), last


class ActorCriticPolicy(nn.Module):
    """
    Actor-critic policy with a swappable feature extractor backbone.

    Architecture
    ------------
    ::

        obs (batch, *obs_shape)
          │
          ▼
        feature_extractor           ← swap for Chronos, NatureCNN, ViT, etc.
          │  (batch, features_dim)
          ▼
        shared MLP  [net_arch]      ← optional shared layers
          │  (batch, shared_dim)
          ├──► actor_head  → logits  (batch, n_actions)
          └──► critic_head → values  (batch,)

    The critic head exists even for REINFORCE (which doesn't use it in the
    loss) so the same policy class is reused by future A2C / PPO without
    modification.

    SB3 compatibility
    -----------------
    The extractor satisfies the
    ``stable_baselines3.common.torch_layers.BaseFeaturesExtractor`` interface,
    so the same extractor classes can be used in SB3 ``policy_kwargs``.

    Args:
        observation_space: Single-env observation space (not batched).
        action_space: Must be ``gym.spaces.Discrete`` (v1).
        net_arch: Hidden layer widths for the shared MLP.  Default: ``[64, 64]``.
        activation_fn: Activation between hidden layers.  Default: ``nn.Tanh``.
        features_extractor_class: Class inheriting ``BaseFeaturesExtractor``.
            Default: ``FlattenExtractor``.
        features_extractor_kwargs: Extra kwargs forwarded to the extractor.

    The policy always receives **batched** tensors — ``(batch, *obs_shape)``
    where ``batch = n_envs`` during rollout collection and
    ``batch = n_steps * n_envs`` during the policy update.
    There are no per-environment loops inside this class.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Discrete,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if net_arch is None:
            net_arch = [64, 64]

        self.observation_space = observation_space
        self.action_space = action_space

        # ── Feature extractor ──────────────────────────────────────────
        fek = features_extractor_kwargs or {}
        self.features_extractor: BaseFeaturesExtractor = features_extractor_class(
            observation_space, **fek
        )
        fd = self.features_extractor.features_dim

        # ── Shared MLP ─────────────────────────────────────────────────
        self.shared_net, shared_dim = _build_mlp(fd, net_arch, activation_fn)

        # ── Heads ──────────────────────────────────────────────────────
        n_actions = int(action_space.n)
        self.actor_head = nn.Linear(shared_dim, n_actions)
        self.critic_head = nn.Linear(shared_dim, 1)

        self.train()

    # ── Core forward ───────────────────────────────────────────────────

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            obs: ``(batch, *obs_shape)``

        Returns:
            ``(logits, values)`` — shapes ``(batch, n_actions)`` and ``(batch,)``.
        """
        features = self.features_extractor(obs)     # (batch, fd)
        shared = self.shared_net(features)           # (batch, shared_dim)
        logits = self.actor_head(shared)             # (batch, n_actions)
        values = self.critic_head(shared).squeeze(-1)  # (batch,)
        return logits, values

    # ── Convenience methods ────────────────────────────────────────────

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        """Return a Categorical distribution for stochastic action sampling."""
        logits, _ = self.forward(obs) # (1) Don't need critic values here; (2) forces the caller to unpack the tuple.
        return Categorical(logits=logits)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities, entropies, and values for a batch of
        (observation, action) pairs.  Used exclusively during ``_update_policy``.

        Args:
            obs:     ``(batch, *obs_shape)``
            actions: ``(batch,)`` int64

        Returns:
            ``(log_probs, entropy, values)`` — all shape ``(batch,)``.
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values

    def predict_actions(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample or greedily select actions for a batch of observations."""
        dist = self.get_distribution(obs)
        return dist.probs.argmax(dim=-1) if deterministic else dist.sample()

    def __repr__(self) -> str:
        return (
            f"ActorCriticPolicy(\n"
            f"  extractor={self.features_extractor},\n"
            f"  shared={self.shared_net},\n"
            f"  actor={self.actor_head},\n"
            f"  critic={self.critic_head}\n)"
        )

"""
On-policy rollout buffer for vectorised environments.

Storage layout
--------------
Every array has shape ``(n_steps, n_envs, …)``.  After collection, the buffer
is flattened to ``(n_steps * n_envs, …)`` for the policy update — there are
no per-environment loops in the update step; the entire batch is processed
at once.

Return computation loops over the **time** axis (reversed), not the env axis.
The env axis is handled by NumPy broadcasting automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """
    Fixed-length rollout buffer for n_envs parallel environments.

    Args:
        n_steps: Time steps collected per environment per update.
        n_envs: Number of parallel environments.
        obs_shape: Shape of a single observation (not batched).
        gamma: Discount factor used in ``compute_returns()``.
        device: Torch device for tensor conversion.

    Typical usage::

        buf = RolloutBuffer(n_steps=2048, n_envs=4, obs_shape=(4,), gamma=0.99, device=device)
        buf.reset()

        for step in range(n_steps):
            buf.add(obs, actions, rewards, log_probs, dones)

        buf.compute_returns(normalize=True)
        batch = buf.to_tensors()   # ready for policy update
    """

    n_steps: int
    n_envs: int
    obs_shape: tuple
    gamma: float
    device: torch.device

    # ── Storage (filled by add()) ──────────────────────────────────────
    observations: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)
    _pos: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all stored data and reset the write pointer."""
        shape_2d = (self.n_steps, self.n_envs)
        self.observations = np.zeros((*shape_2d, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros(shape_2d, dtype=np.int64)
        self.rewards = np.zeros(shape_2d, dtype=np.float32)
        self.log_probs = np.zeros(shape_2d, dtype=np.float32)
        self.dones = np.zeros(shape_2d, dtype=np.float32)
        self.returns = np.zeros(shape_2d, dtype=np.float32)
        self._pos = 0

    def add(
        self,
        obs: np.ndarray,        # (n_envs, *obs_shape)
        actions: np.ndarray,    # (n_envs,)
        rewards: np.ndarray,    # (n_envs,)
        log_probs: np.ndarray,  # (n_envs,)
        dones: np.ndarray,      # (n_envs,) — 1.0 at episode boundaries
    ) -> None:
        assert self._pos < self.n_steps, "Buffer is full; call reset() first."
        self.observations[self._pos] = obs
        self.actions[self._pos] = actions
        self.rewards[self._pos] = rewards
        self.log_probs[self._pos] = log_probs
        self.dones[self._pos] = dones
        self._pos += 1

    @property
    def is_full(self) -> bool:
        return self._pos == self.n_steps

    # ── Return computation ─────────────────────────────────────────────

    def compute_returns(self, normalize: bool = True) -> None:
        """
        Compute discounted returns respecting episode boundaries.

        Iterates over the *time* axis in reverse — the *env* axis is handled
        by NumPy broadcasting (no Python loop over environments).

        When ``dones[t, e] == 1`` the episode for environment ``e`` ended at
        step ``t``; the cumulative return resets to 0 before adding
        ``rewards[t]``.

        Args:
            normalize: Z-score normalise returns across the full buffer
                before the policy update.  Substantially reduces gradient
                variance.  Default: ``True``.
        """
        future = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            future = self.rewards[t] + self.gamma * future * (1.0 - self.dones[t])
            self.returns[t] = future

        if normalize:
            flat = self.returns.ravel()
            self.returns = (self.returns - flat.mean()) / (flat.std() + 1e-8)

    # ── Tensor conversion ──────────────────────────────────────────────

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Flatten ``(n_steps, n_envs, …)`` → ``(n_steps * n_envs, …)`` and
        return PyTorch tensors on ``self.device``.

        Keys: ``observations``, ``actions``, ``log_probs``, ``returns``.
        """
        n = self.n_steps * self.n_envs
        return {
            "observations": torch.from_numpy(
                self.observations.reshape(n, *self.obs_shape)
            ).to(self.device),
            "actions": torch.from_numpy(self.actions.reshape(n)).to(self.device),
            "log_probs": torch.from_numpy(self.log_probs.reshape(n)).to(self.device),
            "returns": torch.from_numpy(self.returns.reshape(n)).to(self.device),
        }

    # ── Episode tracking (for callbacks / logging) ─────────────────────

    def episode_info(self) -> Dict:
        """
        Scan the done flags to extract completed episode rewards.

        Only used for logging and callbacks — not for the policy update.

        Returns:
            ``{"n_episodes", "episode_rewards", "mean_episode_reward"}``.
        """
        # dones: (n_steps, n_envs)
        # Work column-wise with numpy; the Python loop is over log events only.
        episode_rewards: List[float] = []
        running = np.zeros(self.n_envs, dtype=np.float32)

        for t in range(self.n_steps):
            running += self.rewards[t]           # (n_envs,) broadcast add
            finished = self.dones[t] > 0.5       # boolean mask, shape (n_envs,)
            for ep_r in running[finished]:
                episode_rewards.append(float(ep_r))
            running[finished] = 0.0              # reset finished envs

        return {
            "n_episodes": len(episode_rewards),
            "episode_rewards": episode_rewards,
            "mean_episode_reward": (
                float(np.mean(episode_rewards)) if episode_rewards else 0.0
            ),
        }

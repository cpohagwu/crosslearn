from __future__ import annotations

from typing import Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from crosslearn.agents.base import BaseAgent
from crosslearn.buffers import RolloutBuffer
from crosslearn.extractors.base import BaseFeaturesExtractor
from crosslearn.extractors.flatten import FlattenExtractor
from crosslearn.loggers import BaseLogger
from crosslearn.policies.actor_critic import ActorCriticPolicy
from crosslearn.registry import register_agent


@register_agent("reinforce")
class REINFORCE(BaseAgent):
    """
    REINFORCE — Williams (1992) policy gradient, vectorised.

    Collection
    ----------
    ``n_steps`` steps are collected from all ``n_envs`` environments **in
    parallel**.  The policy forward pass always receives a batched tensor of
    shape ``(n_envs, *obs_shape)`` — there are no per-environment Python loops
    inside the policy or the rollout collection.

    Total transitions per update: ``n_steps × n_envs``.

    Episode boundaries
    ------------------
    Done flags from the vectorised env are stored in the rollout buffer.
    Return computation (``RolloutBuffer.compute_returns()``) resets the
    cumulative sum at each done=1 step using NumPy broadcasting across
    the env axis — only the time axis is looped.

    Truncated episodes (time-limit resets) are treated as terminal.
    A critic-bootstrapped variant (A2C) is the natural next step.
    The Critic is unused in the loss (we only use the actor's gradients), 
    but present for A2C/PPO reuse.

    Args:
        env: Any env accepted by ``make_vec_env()``.
        n_envs: Parallel environments.  Ignored if ``env`` is a ``VectorEnv``.
        n_steps: Steps collected per environment per update.  With
            ``n_envs=4`` and ``n_steps=512`` each update sees 2048 transitions.
            Default: 2048.
        normalize_returns: Z-score normalise discounted returns before the
            policy loss.  Equivalent to a mean-baseline REINFORCE.
            Default: ``True``.
        entropy_coeff: Entropy bonus weight to prevent premature collapse.
            Tune in ``[0.005, 0.05]``. Default: 0.01.
        max_grad_norm: Gradient clipping norm. Default: 0.5.
        features_extractor_class: Backbone extractor.
            Default: ``FlattenExtractor`` (flat MLP).
            Swap for ``ChronosExtractor``, ``NatureCNNExtractor``, or any SB3
            ``BaseFeaturesExtractor`` subclass.
        features_extractor_kwargs: Extra kwargs for the extractor.
        policy_kwargs: Forwarded to ``ActorCriticPolicy``.
            Keys: ``net_arch`` (list[int]), ``activation_fn`` (nn.Module class).
        gamma: Discount factor. Default: 0.99.
        learning_rate: Adam LR. Default: 3e-4.
        optimizer_class: Optimiser class. Default: ``torch.optim.Adam``.
        lr_scheduler_class: Optional LR scheduler class.
        lr_scheduler_kwargs: Kwargs for the scheduler.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        use_async_env: Forwarded to ``make_vec_env()`` when ``env`` is not
            already vectorized. Use for slower environments where multi-process
            stepping can hide CPU-side rollout latency.
        logger: Optional logger.
        verbose: 0 = silent, 1 = info, 2 = debug every update.
        seed: Random seed.

    Example — CartPole with callbacks::

        import gymnasium as gym
        from crosslearn import REINFORCE
        from crosslearn.callbacks import EpisodeSolvedCallback, ProgressBarCallback
        from crosslearn.envs.utils import make_vec_env

        vec = make_vec_env("CartPole-v1", n_envs=4)
        agent = REINFORCE(vec, n_steps=512, normalize_returns=True)
        agent.learn(
            total_timesteps=500_000,
            callbacks=[
                EpisodeSolvedCallback(reward_threshold=475.0),
                ProgressBarCallback(),
            ],
        )

    Example — Trading windows with a Chronos backbone::

        from crosslearn.extractors import ChronosExtractor
        from crosslearn.envs.utils import make_vec_env

        vec = make_vec_env(
            lambda: MyTradingEnv(window_size=20),
            n_envs=8,
        )
        agent = REINFORCE(
            vec,
            n_steps=256,
            features_extractor_class=ChronosExtractor,
            features_extractor_kwargs=dict(
                feature_names=["Open", "High", "Low", "Close", "Volume"],
                selected_columns=["Close", "Volume"],
            ),
        )
        agent.learn(total_timesteps=1_000_000)
    """

    def __init__(
        self,
        env,
        n_envs: int = 1,
        n_steps: int = 2048,
        normalize_returns: bool = True,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        policy_kwargs: Optional[Dict] = None,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr_scheduler_class: Optional[Type] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        device: str = "auto",
        use_async_env: bool = False,
        logger: Optional[BaseLogger] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        # ── Store REINFORCE-specific attrs before super().__init__ ─────
        # (super calls _build_policy / _build_optimizer immediately)
        self.n_steps = n_steps
        self.normalize_returns = normalize_returns
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.optimizer_class = optimizer_class
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        super().__init__(
            env=env,
            n_envs=n_envs,
            policy_kwargs=policy_kwargs,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            gamma=gamma,
            learning_rate=learning_rate,
            device=device,
            use_async_env=use_async_env,
            logger=logger,
            verbose=verbose,
            seed=seed,
        )

        # Build rollout buffer now that n_envs and obs_shape are known
        self._buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_shape=self.observation_space.shape,
            gamma=self.gamma,
            device=self.device,
        )

        # Optional LR scheduler (built after super creates self.optimizer)
        self.lr_scheduler = None
        if self.lr_scheduler_class is not None:
            self.lr_scheduler = self.lr_scheduler_class(
                self.optimizer, **self.lr_scheduler_kwargs
            )

    # ── BaseAgent abstract implementations ────────────────────────────

    def _build_policy(self) -> nn.Module:
        return ActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self._resolve_features_extractor_kwargs(),
            **self.policy_kwargs,
        ).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer_class(self.policy.parameters(), lr=self.learning_rate)

    def _collect_rollout(self) -> Dict:
        """
        Collect ``n_steps`` steps from all ``n_envs`` environments in parallel.

        The policy forward pass receives a batched tensor ``(n_envs, *obs_shape)``
        — no Python loop over environments.  Only the time-step axis is looped
        (sequential dependency: step t+1 depends on env state after step t).

        Returns dict with keys expected by ``_update_policy()`` and callbacks.
        """
        self._buffer.reset()
        obs = self._last_obs   # (n_envs, *obs_shape)
        episode_rewards: List[float] = []

        for _ in range(self.n_steps):
            obs_t = torch.from_numpy(obs).float().to(self.device)

            # Policy forward: (n_envs, *obs_shape) → dist over (n_envs,) actions
            with torch.no_grad():
                dist = self.policy.get_distribution(obs_t)
                actions = dist.sample()           # (n_envs,)
                log_probs = dist.log_prob(actions)  # (n_envs,)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, _ = self.env.step(actions_np)
            rewards = rewards.astype(np.float32)
            dones = (terminated | truncated).astype(np.float32) 

            self._episode_returns += rewards
            finished = dones > 0.5 # bool mask for envs that finished this step
            for ep_r in self._episode_returns[finished]: 
                episode_rewards.append(float(ep_r)) # add finished episode returns to list
            self._episode_returns[finished] = 0.0

            self._buffer.add(
                obs=obs,
                actions=actions_np,
                rewards=rewards,
                log_probs=log_probs.cpu().numpy(),
                dones=dones,
            )
            obs = next_obs

        self._last_obs = obs
        self._buffer.compute_returns(normalize=self.normalize_returns)

        mean_episode_reward = (
            float(np.mean(episode_rewards)) if episode_rewards else 0.0
        )
        return {
            "n_steps": self.n_steps * self.n_envs,
            "n_episodes": len(episode_rewards),
            "episode_rewards": episode_rewards,
            "mean_episode_reward": mean_episode_reward,
        }

    def _update_policy(self, rollout: Dict) -> Dict:
        """
        Single gradient step over the entire flattened rollout buffer.

        Loss
        ----
        .. math::
            \\mathcal{L} = -\\mathbb{E}[\\log\\pi(a|s) \\cdot G_t]
                          - \\alpha\\,H[\\pi(\\cdot|s)]

        The full ``(n_steps × n_envs)`` batch is processed in **one**
        ``evaluate_actions`` call — no per-step or per-env loops.
        """
        batch = self._buffer.to_tensors()
        obs = batch["observations"]
        actions = batch["actions"]
        returns = batch["returns"]

        # One batched forward pass: (n_steps * n_envs, *obs_shape)
        log_probs, entropy, _ = self.policy.evaluate_actions(obs, actions)

        policy_loss = -(log_probs * returns).mean()
        entropy_loss = -self.entropy_coeff * entropy.mean()
        total_loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "train/policy_loss": policy_loss.item(),
            "train/entropy_loss": entropy_loss.item(),
            "train/total_loss": total_loss.item(),
            "train/entropy": entropy.mean().item(),
            "train/mean_episode_reward": rollout.get("mean_episode_reward", 0.0),
            "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    def _predict_action(
        self, obs_t: torch.Tensor, deterministic: bool
    ) -> np.ndarray:
        return self.policy.predict_actions(obs_t, deterministic).cpu().numpy()

    def _get_hyperparams(self) -> Dict:
        return {
            **super()._get_hyperparams(),
            "n_steps": self.n_steps,
            "normalize_returns": self.normalize_returns,
            "entropy_coeff": self.entropy_coeff,
            "max_grad_norm": self.max_grad_norm,
        }

    def _default_hyperparams(self) -> Dict:
        return {
            **super()._default_hyperparams(),
            "n_steps": 2048,
            "normalize_returns": True,
            "entropy_coeff": 0.01,
            "max_grad_norm": 0.5,
            "optimizer_class": torch.optim.Adam,
            "lr_scheduler_class": None,
            "lr_scheduler_kwargs": {},
        }

    def __repr__(self) -> str:
        return (
            f"REINFORCE(n_envs={self.n_envs}, n_steps={self.n_steps}, "
            f"gamma={self.gamma}, lr={self.learning_rate}, "
            f"normalize_returns={self.normalize_returns})"
        )

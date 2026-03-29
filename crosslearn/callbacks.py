"""
Training callbacks.  Callbacks are passed to ``agent.learn()`` (not the
constructor) and are re-initialised at the start of each ``learn()`` call.

Hook execution order
--------------------
``on_training_start``
  └─ (rollout loop)
       on_rollout_end  ← called after every policy update
       on_best_model   ← called when EpisodeSolvedCallback fires
``on_training_end``

Design note on stop signals
---------------------------
Callbacks communicate back to the training loop by setting
``agent._stop_training = True``.  The ``learn()`` loop checks this flag at
the end of every rollout iteration.
"""
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from crosslearn.agents.base import BaseAgent
    from crosslearn.loggers import BaseLogger


class BaseCallback(ABC):
    """
    Abstract base class for training callbacks.

    All hooks are no-ops by default — override only what you need.
    """

    def on_training_start(
        self,
        agent: "BaseAgent",
        total_timesteps: Optional[int],
        total_episodes: Optional[int],
    ) -> None:
        """Called once before the training loop begins."""

    def on_rollout_end(
        self, rollout: dict, metrics: dict, agent: "BaseAgent"
    ) -> None:
        """Called after every policy update (one gradient step)."""

    def on_best_model(self, mean_reward: float, agent: "BaseAgent") -> None:
        """Called when a new best mean reward is recorded."""

    def on_training_end(self, agent: "BaseAgent") -> None:
        """Called once after the training loop exits."""


class CallbackList(BaseCallback):
    """Internal: chains multiple callbacks and dispatches every hook."""

    def __init__(self, callbacks: List[BaseCallback]) -> None:
        self.callbacks = callbacks

    def on_training_start(self, agent, total_timesteps, total_episodes):
        for cb in self.callbacks:
            cb.on_training_start(agent, total_timesteps, total_episodes)

    def on_rollout_end(self, rollout, metrics, agent):
        for cb in self.callbacks:
            cb.on_rollout_end(rollout, metrics, agent)

    def on_best_model(self, mean_reward, agent):
        for cb in self.callbacks:
            cb.on_best_model(mean_reward, agent)

    def on_training_end(self, agent):
        for cb in self.callbacks:
            cb.on_training_end(agent)


class EpisodeSolvedCallback(BaseCallback):
    """
    Stop training when the rolling mean episode reward meets the threshold.

    Args:
        reward_threshold: Mean reward required to declare the task solved.
        n_episodes: Rolling window size. Default: 100 (standard benchmark).
        verbose: 0 = silent, 1 = print when solved.

    Example::

        agent.learn(
            total_timesteps=500_000,
            callbacks=[EpisodeSolvedCallback(reward_threshold=475.0)],
        )
    """

    def __init__(
        self,
        reward_threshold: float,
        n_episodes: int = 100,
        verbose: int = 1,
    ) -> None:
        self.reward_threshold = reward_threshold
        self.n_episodes = n_episodes
        self.verbose = verbose

    def on_rollout_end(self, rollout, metrics, agent):
        if len(agent._rolling_episode_rewards) < agent._rolling_window:
            return

        mean_r = metrics.get(
            "train/rolling_mean_episode_reward",
            agent._rolling_mean_episode_reward,
        )
        if mean_r >= self.reward_threshold:
            agent._stop_training = True
            if self.verbose:
                print(
                    f"\n  [EpisodeSolvedCallback] Solved!  "
                    f"Rolling mean over {self.n_episodes} episodes: "
                    f"{mean_r:.2f} ≥ {self.reward_threshold}"
                )
            agent.callbacks.on_best_model(mean_r, agent)


class BestModelCallback(BaseCallback):
    """
    Save the model whenever a new best mean reward is reported.

    Triggered by ``on_best_model``, which is fired either by
    ``EpisodeSolvedCallback`` or by the eval loop in ``BaseAgent.learn()``.

    Args:
        save_path: Full path for the checkpoint, e.g. ``"./models/best.pt"``.
        logger: Optional logger.  If it supports ``log_artifact()`` (WandbLogger),
            the checkpoint is also uploaded as an artifact.
        verbose: 0 = silent, 1 = print on each save.
    """

    def __init__(
        self,
        save_path: str,
        logger: Optional["BaseLogger"] = None,
        verbose: int = 1,
    ) -> None:
        self.save_path = Path(save_path)
        self.logger = logger
        self.verbose = verbose

    def on_best_model(self, mean_reward, agent):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(self.save_path)
        if self.verbose:
            print(
                f"  [BestModelCallback] New best {mean_reward:.2f}  →  {self.save_path}"
            )
        if self.logger is not None:
            spec = getattr(getattr(agent.env, "unwrapped", agent.env), "spec", None)
            name = (spec.id.replace("/", "-") if spec else "env") + "_best"
            self.logger.log_artifact(str(self.save_path), name)


class CheckpointCallback(BaseCallback):
    """
    Save a checkpoint every ``save_freq`` policy updates.

    Args:
        save_freq: Save every N rollout updates.
        save_path: Directory for checkpoint files.
        name_prefix: Filename prefix. Default: ``"checkpoint"``.
        verbose: 0 = silent, 1 = print on each save.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "checkpoint",
        verbose: int = 1,
    ) -> None:
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.verbose = verbose
        self._n_calls = 0

    def on_rollout_end(self, rollout, metrics, agent):
        self._n_calls += 1
        if self._n_calls % self.save_freq == 0:
            self.save_path.mkdir(parents=True, exist_ok=True)
            path = self.save_path / f"{self.name_prefix}_{agent._n_timesteps}_steps.pt"
            agent.save(path)
            if self.verbose:
                print(f"  [CheckpointCallback] Saved → {path}")


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training if mean reward does not improve by ``min_delta`` for
    ``patience`` consecutive ``on_best_model`` calls.

    Args:
        patience: Evaluations without improvement before stopping. Default: 10.
        min_delta: Minimum improvement required. Default: 0.01.
        verbose: 0 = silent, 1 = print when triggered.
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 0.01, verbose: int = 1
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self._best = -float("inf")
        self._no_improve = 0

    def on_best_model(self, mean_reward, agent):
        if mean_reward - self._best > self.min_delta:
            self._best = mean_reward
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self._no_improve >= self.patience:
                agent._stop_training = True
                if self.verbose:
                    print(
                        f"  [EarlyStoppingCallback] No improvement for "
                        f"{self.patience} evals. Stopping."
                    )


class ProgressBarCallback(BaseCallback):
    """
    ``tqdm`` progress bar that tracks training progress.

    Displays total steps or episodes (whichever is the termination criterion),
    mean episode reward, and current loss.

    Requires ``tqdm``:  ``pip install tqdm``.

    Example::

        agent.learn(total_timesteps=500_000, callbacks=[ProgressBarCallback()])
    """

    def __init__(self) -> None:
        self._pbar = None
        self._use_steps: bool = True

    def on_training_start(self, agent, total_timesteps, total_episodes):
        try:
            from tqdm import tqdm
        except ImportError:
            raise ImportError("ProgressBarCallback requires tqdm: pip install tqdm")

        self._use_steps = total_timesteps is not None
        total = total_timesteps or total_episodes
        unit = "steps" if self._use_steps else "eps"
        self._pbar = tqdm(total=total, unit=unit, dynamic_ncols=True)

    def on_rollout_end(self, rollout, metrics, agent):
        if self._pbar is None:
            return
        delta = rollout["n_steps"] if self._use_steps else rollout["n_episodes"]
        self._pbar.update(delta)
        postfix: dict = {}
        if rollout.get("n_episodes", 0) > 0:
            postfix["ep_reward"] = f"{rollout['mean_episode_reward']:.2f}"
        if "train/total_loss" in metrics:
            postfix["loss"] = f"{metrics['train/total_loss']:.4f}"
        postfix["n_eps"] = agent._n_episodes
        self._pbar.set_postfix(postfix)

    def on_training_end(self, agent):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

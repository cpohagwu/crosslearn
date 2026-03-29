from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from crosslearn.callbacks import BaseCallback, CallbackList, EpisodeSolvedCallback
from crosslearn.envs.utils import make_vec_env
from crosslearn.extractors.base import BaseFeaturesExtractor
from crosslearn.extractors.flatten import FlattenExtractor
from crosslearn.loggers import BaseLogger


class BaseAgent(ABC):
    """
    Abstract base class for package agents.

    All environments are normalised to a ``gym.vector.VectorEnv`` on entry.
    Passing a single ``gym.Env`` or ``gym.make_vec()`` output both work; the
    policy always sees batched observations of shape ``(n_envs, *obs_shape)``.

    Callbacks are passed to ``learn()`` (not the constructor) so the same
    agent instance can be trained with different callback sets across calls.

    Args:
        env: Single env, vector env, string env-id, or callable factory.
            See ``make_vec_env`` for full details.
        n_envs: Only used when ``env`` is a string or callable.  Ignored if
            ``env`` is already a ``VectorEnv``.
        policy_kwargs: Forwarded to the policy constructor.  Keys:
            ``net_arch``, ``activation_fn``,
            ``features_extractor_class``, ``features_extractor_kwargs``.
        features_extractor_class: Feature extractor class.
            Default: ``FlattenExtractor``.
        features_extractor_kwargs: Extra kwargs for the extractor.
        gamma: Discount factor. Default: 0.99.
        learning_rate: Optimiser initial LR. Default: 3e-4.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        logger: Optional ``BaseLogger`` instance.
        verbose: 0 = silent, 1 = info, 2 = debug (prints every metric).
        seed: Optional random seed.
    """

    def __init__(
        self,
        env: Union[str, gym.Env, gym.vector.VectorEnv],
        n_envs: int = 1,
        policy_kwargs: Optional[Dict] = None,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        device: str = "auto",
        logger: Optional[BaseLogger] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        # ── Vectorise environment ──────────────────────────────────────
        self.env: gym.vector.VectorEnv = make_vec_env(env, n_envs=n_envs)
        self.n_envs: int = self.env.num_envs

        # Single-env spaces used for policy construction
        self.observation_space: gym.Space = self.env.single_observation_space
        self.action_space: gym.Space = self.env.single_action_space

        # ── Hyperparameters ────────────────────────────────────────────
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.seed = seed
        self.logger = logger
        self.policy_kwargs = policy_kwargs or {}
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs or {}

        # ── Device ────────────────────────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            if device == "auto" else device
        )

        # ── Reproducibility ────────────────────────────────────────────
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ── Training state ─────────────────────────────────────────────
        self._n_timesteps: int = 0
        self._n_episodes: int = 0
        self._n_updates: int = 0
        self._best_mean_reward: float = -np.inf
        self._stop_training: bool = False
        self._last_obs: Optional[np.ndarray] = None  # maintained between rollouts
        self._episode_returns: np.ndarray = np.zeros(self.n_envs, dtype=np.float32)
        self._rolling_window: int = 100
        self._rolling_episode_rewards = deque(maxlen=self._rolling_window)
        self._rolling_mean_episode_reward: float = 0.0

        # Internal callbacks handle (set per learn() call)
        self.callbacks: CallbackList = CallbackList([])

        # ── Build policy & optimiser ───────────────────────────────────
        # NOTE: subclasses must set all algorithm-specific attributes
        # BEFORE calling super().__init__() because _build_policy() and
        # _build_optimizer() are called here.
        self.policy: nn.Module = self._build_policy()
        self.optimizer: torch.optim.Optimizer = self._build_optimizer()

        if self.verbose >= 1:
            print(
                f"{self.__class__.__name__} | n_envs={self.n_envs} | "
                f"device={self.device} | obs={self.observation_space.shape} | "
                f"actions={self.action_space.n}"
            )
            if self.verbose >= 2:
                print(self.policy)

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def _build_policy(self) -> nn.Module: ...

    @abstractmethod
    def _build_optimizer(self) -> torch.optim.Optimizer: ...

    @abstractmethod
    def _collect_rollout(self) -> Dict: ...

    @abstractmethod
    def _update_policy(self, rollout: Dict) -> Dict: ...

    @abstractmethod
    def _predict_action(self, obs_t: torch.Tensor, deterministic: bool) -> np.ndarray: ...

    # ── Public API ─────────────────────────────────────────────────────

    def learn(
        self,
        total_timesteps: Optional[int] = None,
        total_episodes: Optional[int] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        reset_num_timesteps: bool = True,
    ) -> "BaseAgent":
        """
        Train the agent.

        Args:
            total_timesteps: Stop after this many environment steps.
                One step = one transition across *all* ``n_envs`` environments.
                Total env interactions = ``total_timesteps``.
            total_episodes: Stop after this many completed episodes
                (summed across all parallel environments).
            callbacks: List of ``BaseCallback`` instances.  Applied for this
                ``learn()`` call only; they are re-initialised on each call.
            eval_env: Optional single environment for periodic greedy eval.
                When provided, ``on_best_model`` fires on new best.
            eval_freq: Evaluate every N *total* timesteps.
            n_eval_episodes: Episodes per evaluation.
            reset_num_timesteps: Reset ``_n_timesteps`` / ``_n_episodes``
                counters.  Set ``False`` to continue a previous run.

        Returns:
            ``self`` — enables chaining::

                agent = REINFORCE(env).learn(total_timesteps=500_000,
                                             callbacks=[ProgressBarCallback()])
        """
        if total_timesteps is None and total_episodes is None:
            raise ValueError("Provide total_timesteps or total_episodes (or both).")

        if reset_num_timesteps:
            self._n_timesteps = 0
            self._n_episodes = 0
            self._n_updates = 0
            self._best_mean_reward = -np.inf
            self._stop_training = False
            self._episode_returns[:] = 0.0

        # Initialise last obs for rollout continuity
        if self._last_obs is None or reset_num_timesteps:
            obs, _ = self.env.reset(seed=self.seed)
            self._last_obs = obs

        # ── Set up callbacks for this run ──────────────────────────────
        cb_list = callbacks or []
        self.callbacks = CallbackList(cb_list)
        self._rolling_window = 100
        for cb in cb_list:
            if isinstance(cb, EpisodeSolvedCallback):
                self._rolling_window = cb.n_episodes
                break
        self._rolling_episode_rewards = deque(maxlen=self._rolling_window)
        self._rolling_mean_episode_reward = 0.0
        self.callbacks.on_training_start(self, total_timesteps, total_episodes)
        last_eval_step = -eval_freq

        if self.logger:
            run_config = self._build_run_config(
                total_timesteps=total_timesteps,
                total_episodes=total_episodes,
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                reset_num_timesteps=reset_num_timesteps,
            )
            self.logger.log_config(run_config)

        if self.verbose >= 1:
            parts = []
            if total_timesteps:
                parts.append(f"total_timesteps={total_timesteps:,}")
            if total_episodes:
                parts.append(f"total_episodes={total_episodes:,}")
            print(f"Training {self.__class__.__name__} | " + " / ".join(parts))

        # ── Main training loop ─────────────────────────────────────────
        while self._should_continue(total_timesteps, total_episodes):

            rollout = self._collect_rollout()
            self._n_timesteps += rollout["n_steps"]
            self._n_episodes += rollout["n_episodes"]
            self._n_updates += 1

            metrics = self._update_policy(rollout)
            metrics.update({
                "train/n_timesteps": self._n_timesteps,
                "train/n_episodes": self._n_episodes,
                "train/n_updates": self._n_updates,
            })
            episode_rewards = rollout.get("episode_rewards", [])
            if episode_rewards:
                self._rolling_episode_rewards.extend(episode_rewards)
            if len(self._rolling_episode_rewards) >= self._rolling_window:
                self._rolling_mean_episode_reward = float(
                    np.mean(self._rolling_episode_rewards)
                )
            else:
                self._rolling_mean_episode_reward = 0.0
            metrics["train/rolling_mean_episode_reward"] = (
                self._rolling_mean_episode_reward
            )

            if self.logger:
                self.logger.log(metrics, step=self._n_timesteps)

            if self.verbose >= 2:
                ep_r = rollout.get("mean_episode_reward", 0.0)
                loss = metrics.get("train/total_loss", 0.0)
                print(
                    f"  update={self._n_updates:>5} | "
                    f"steps={self._n_timesteps:>10,} | "
                    f"episodes={self._n_episodes:>6} | "
                    f"ep_reward={ep_r:>8.2f} | "
                    f"rolling_mean_ep_reward={self._rolling_mean_episode_reward:>8.2f} | "
                    f"loss={loss:>10.4f}"
                )

            self.callbacks.on_rollout_end(rollout, metrics, self)

            # Periodic evaluation on a separate env
            if eval_env is not None:
                steps_since = self._n_timesteps - last_eval_step
                if steps_since >= eval_freq:
                    last_eval_step = self._n_timesteps
                    mean_r = self._evaluate(eval_env, n_eval_episodes)
                    eval_metrics = {"eval/mean_reward": mean_r}
                    if self.logger:
                        self.logger.log(eval_metrics, step=self._n_timesteps)
                    if self.verbose >= 1:
                        print(
                            f"  [Eval] steps={self._n_timesteps:,}  "
                            f"mean_reward={mean_r:.2f}"
                            + (f"  (best: {self._best_mean_reward:.2f})"
                               if self._best_mean_reward > -np.inf else "")
                        )
                    if mean_r > self._best_mean_reward:
                        self._best_mean_reward = mean_r
                        self.callbacks.on_best_model(mean_r, self)

            if self._stop_training:
                if self.verbose >= 1:
                    print("  Training stopped by callback.")
                break

        # ── Finalise ───────────────────────────────────────────────────
        if self.verbose >= 1:
            print(
                f"Done | steps={self._n_timesteps:,} | "
                f"episodes={self._n_episodes} | updates={self._n_updates}"
            )
        self.callbacks.on_training_end(self)
        if self.logger:
            self.logger.close()

        return self

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Select actions for a batch or single observation.

        Args:
            obs: Shape ``(*obs_shape,)`` for a single obs or
                 ``(n, *obs_shape)`` for a batch.
            deterministic: Greedy if ``True`` (default), stochastic if ``False``.

        Returns:
            Integer action(s) as a numpy array.
        """
        single = obs.ndim == len(self.observation_space.shape)
        if single:
            obs = obs[np.newaxis]
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            actions = self._predict_action(obs_t, deterministic)
        return actions[0] if single else actions

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "hyperparams": self._get_hyperparams(),
                "n_timesteps": self._n_timesteps,
                "n_episodes": self._n_episodes,
                "best_mean_reward": self._best_mean_reward,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> "BaseAgent":
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._n_timesteps = ckpt.get("n_timesteps", 0)
        self._n_episodes = ckpt.get("n_episodes", 0)
        self._best_mean_reward = ckpt.get("best_mean_reward", -np.inf)
        if self.verbose >= 1:
            print(f"Loaded ← {path}  (step={self._n_timesteps:,})")
        return self

    @classmethod
    def load_from_path(
        cls, path: Union[str, Path], env, **kwargs
    ) -> "BaseAgent":
        """Create agent and immediately restore a checkpoint."""
        agent = cls(env, verbose=0, **kwargs)
        agent.load(path)
        return agent

    # ── Internal helpers ───────────────────────────────────────────────

    def _evaluate(self, env: gym.Env, n_episodes: int) -> float:
        total = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(int(action))
                total += float(r)
                done = term or trunc
        return total / n_episodes

    def _should_continue(
        self,
        total_timesteps: Optional[int],
        total_episodes: Optional[int],
    ) -> bool:
        if self._stop_training:
            return False
        if total_timesteps is not None and self._n_timesteps >= total_timesteps:
            return False
        if total_episodes is not None and self._n_episodes >= total_episodes:
            return False
        return True

    def _get_hyperparams(self) -> Dict:
        return {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "n_envs": self.n_envs,
            "policy_kwargs": self.policy_kwargs,
        }

    def _default_hyperparams(self) -> Dict:
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return {
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "n_envs": 1,
            "policy_kwargs": {},
            "features_extractor_class": FlattenExtractor,
            "features_extractor_kwargs": {},
            "seed": None,
            "verbose": 1,
            "device": str(resolved_device),
        }

    def _default_learn_params(self) -> Dict:
        return {
            "total_timesteps": None,
            "total_episodes": None,
            "eval_freq": 10_000,
            "n_eval_episodes": 5,
            "reset_num_timesteps": True,
            "eval_env_provided": False,
            "eval_env_id": None,
        }

    def _build_run_config(
        self,
        total_timesteps: Optional[int],
        total_episodes: Optional[int],
        eval_env: Optional[gym.Env],
        eval_freq: int,
        n_eval_episodes: int,
        reset_num_timesteps: bool,
    ) -> Dict[str, Any]:
        env_id = self._resolve_env_id(self.env)
        eval_env_id = self._resolve_env_id(eval_env) if eval_env is not None else None

        raw_hyperparams: Dict[str, Any] = {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "n_envs": self.n_envs,
            "policy_kwargs": self.policy_kwargs,
            "features_extractor_class": self.features_extractor_class,
            "features_extractor_kwargs": self.features_extractor_kwargs,
            "seed": self.seed,
            "verbose": self.verbose,
            "device": str(self.device),
        }
        for name in (
            "n_steps",
            "normalize_returns",
            "entropy_coeff",
            "max_grad_norm",
            "optimizer_class",
            "lr_scheduler_class",
            "lr_scheduler_kwargs",
        ):
            if hasattr(self, name):
                raw_hyperparams[name] = getattr(self, name)

        raw_learn: Dict[str, Any] = {
            "total_timesteps": total_timesteps,
            "total_episodes": total_episodes,
            "eval_freq": eval_freq,
            "n_eval_episodes": n_eval_episodes,
            "reset_num_timesteps": reset_num_timesteps,
            "eval_env_provided": eval_env is not None,
            "eval_env_id": eval_env_id,
        }

        hyper_defaults = self._default_hyperparams()
        learn_defaults = self._default_learn_params()

        hyperparams_source = self._build_source_map(
            raw_hyperparams, hyper_defaults
        )
        learn_source = self._build_source_map(
            raw_learn,
            learn_defaults,
            derived_keys={"eval_env_provided", "eval_env_id"},
        )

        return {
            "algorithm": self.__class__.__name__,
            "env_id": env_id,
            "hyperparams": self._serialize_value(raw_hyperparams),
            "hyperparams_source": hyperparams_source,
            "learn": self._serialize_value(raw_learn),
            "learn_source": learn_source,
        }

    @staticmethod
    def _resolve_env_id(env: Optional[gym.Env]) -> Optional[str]:
        if env is None:
            return None
        base = getattr(env, "unwrapped", env)
        spec = getattr(base, "spec", None)
        if spec is not None:
            env_id = getattr(spec, "id", None)
            if env_id:
                return env_id
        envs = getattr(env, "envs", None)
        if envs:
            return BaseAgent._resolve_env_id(envs[0])
        return None

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        try:
            return left == right
        except Exception:
            return False

    def _build_source_map(
        self,
        values: Dict[str, Any],
        defaults: Dict[str, Any],
        derived_keys: Optional[set] = None,
    ) -> Dict[str, str]:
        derived_keys = derived_keys or set()
        source: Dict[str, str] = {}
        for key, value in values.items():
            if key in derived_keys:
                source[key] = "derived"
                continue
            if key not in defaults:
                source[key] = "specified"
                continue
            source[key] = (
                "default"
                if self._values_equal(value, defaults[key])
                else "specified"
            )
        return source

    def _serialize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_value(v) for v in value]
        if inspect.isclass(value):
            return f"{value.__module__}.{value.__qualname__}"
        if callable(value):
            return f"{value.__module__}.{value.__qualname__}"
        return str(value)

    @abstractmethod
    def __repr__(self) -> str: ...

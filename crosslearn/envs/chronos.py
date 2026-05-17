from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import utils as space_utils

from crosslearn.extractors.chronos import ChronosEmbedder


class WalkForwardChronosWrapper(
    gym.Wrapper[np.ndarray, Any, np.ndarray, Any],
    gym.utils.RecordConstructorArgs,
):
    """Online Chronos wrapper for envs that emit one observation at a time.

    The wrapper collects raw observations into a walk-forward history, embeds
    that history with Chronos, and returns the pooled Chronos vector as the
    agent-visible observation.

    Args:
        env: Wrapped environment. Its observation space must be flattenable by
            Gymnasium.
        lookback: Rolling history length once enough observations have been
            collected.
        feature_names: Optional names for the flattened raw observation
            features. When omitted, all flattened features are embedded.
        selected_columns: Optional subset of ``feature_names`` to embed by name.
        selected_indices: Optional subset of flattened feature positions to
            embed by index.
        min_history: Minimum number of collected observations required before
            emitting a real Chronos embedding. Defaults to ``lookback``.
        warmup_value: Placeholder value returned until ``min_history`` has been
            collected.
        expanding_window: If ``True``, embed the full collected history after
            warmup instead of the last ``lookback`` observations.
        max_history: Optional cap for stored history when ``expanding_window``
            is ``True``. Ignored for rolling windows.
        model_name: Chronos model identifier.
        pooling: Token pooling mode forwarded to ``ChronosEmbedder``.
        device_map: Target device for Chronos.
        dtype: Torch dtype used when loading Chronos.
        cache_size: Optional LRU cache size for repeated online windows.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        lookback: int,
        feature_names: Sequence[str] | None = None,
        selected_columns: Sequence[str] | None = None,
        selected_indices: Sequence[int] | None = None,
        min_history: int | None = None,
        warmup_value: float = 0.0,
        expanding_window: bool = False,
        max_history: int | None = None,
        model_name: str = "amazon/chronos-2",
        pooling: str = "mean",
        device_map: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
        cache_size: int | None = 16_384,
    ) -> None:
        gym.utils.RecordConstructorArgs.__init__(
            self,
            lookback=lookback,
            feature_names=list(feature_names) if feature_names is not None else None,
            selected_columns=(
                list(selected_columns) if selected_columns is not None else None
            ),
            selected_indices=(
                [int(index) for index in selected_indices]
                if selected_indices is not None
                else None
            ),
            min_history=min_history,
            warmup_value=warmup_value,
            expanding_window=expanding_window,
            max_history=max_history,
            model_name=model_name,
            pooling=pooling,
            device_map=device_map,
            dtype=dtype,
            cache_size=cache_size,
        )
        super().__init__(env)

        self.lookback = int(lookback)
        if self.lookback <= 0:
            raise ValueError("lookback must be greater than 0.")

        self.min_history = self.lookback if min_history is None else int(min_history)
        if self.min_history <= 0:
            raise ValueError("min_history must be greater than 0.")

        self.expanding_window = bool(expanding_window)
        self.max_history = None if max_history is None else int(max_history)
        if self.max_history is not None and self.max_history <= 0:
            raise ValueError("max_history must be greater than 0 or None.")
        if (
            self.expanding_window
            and self.max_history is not None
            and self.max_history < self.min_history
        ):
            raise ValueError("max_history must be at least min_history.")

        self.warmup_value = float(warmup_value)
        self.n_features = int(space_utils.flatdim(env.observation_space))
        self.feature_names = list(feature_names) if feature_names is not None else None
        if self.feature_names is not None and len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries, but "
                f"the flattened observation has {self.n_features} features."
            )

        self.embedder = ChronosEmbedder(
            model_name=model_name,
            pooling=pooling,
            feature_names=self.feature_names,
            selected_columns=selected_columns,
            selected_indices=selected_indices,
            device_map=device_map,
            dtype=dtype,
            cache_size=cache_size,
        )

        example_features = self.embedder.embed_windows(
            np.zeros((self.lookback, self.n_features), dtype=np.float32),
            n_features=self.n_features,
            feature_names=self.feature_names,
            as_tensor=False,
        )
        self.embedder.clear_cache()
        self.embedding_dim = int(example_features.shape[-1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.embedding_dim,),
            dtype=np.float32,
        )
        self._history: list[np.ndarray] = []

    def _flatten_observation(self, observation: Any) -> np.ndarray:
        try:
            flattened = space_utils.flatten(self.env.observation_space, observation)
        except Exception:
            flattened = np.asarray(observation, dtype=np.float32).reshape(-1)
        return np.asarray(flattened, dtype=np.float32).reshape(-1)

    def _append_observation(self, observation: Any) -> None:
        flattened = self._flatten_observation(observation)
        if flattened.size != self.n_features:
            raise ValueError(
                f"Expected flattened observation with {self.n_features} features, "
                f"got {flattened.size}."
            )
        self._history.append(flattened.copy())

        if self.expanding_window:
            if self.max_history is not None and len(self._history) > self.max_history:
                self._history = self._history[-self.max_history :]
        else:
            retained_history = max(self.lookback, self.min_history)
            if len(self._history) > retained_history:
                self._history = self._history[-retained_history :]

    def _current_window(self) -> np.ndarray:
        if self.expanding_window:
            return np.stack(self._history, axis=0)
        return np.stack(self._history[-self.lookback :], axis=0)

    def _embedded_observation(self) -> np.ndarray:
        if len(self._history) < self.min_history:
            return np.full(
                (self.embedding_dim,),
                self.warmup_value,
                dtype=np.float32,
            )

        embedding = self.embedder.embed_windows(
            self._current_window(),
            n_features=self.n_features,
            feature_names=self.feature_names,
            as_tensor=False,
        )
        return embedding[0].astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._history = []
        self._append_observation(observation)
        return self._embedded_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._append_observation(observation)
        return self._embedded_observation(), reward, terminated, truncated, info


__all__ = ["WalkForwardChronosWrapper"]

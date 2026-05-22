from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from crosslearn.envs._chronos_observation import ChronosObservationAdapter, ObservationMode
from crosslearn.extractors.chronos import ChronosEmbedder


class WalkForwardChronosWrapper(
    gym.Wrapper[np.ndarray, Any, np.ndarray, Any],
    gym.utils.RecordConstructorArgs,
):
    """Online Chronos wrapper for env-side Chronos features.

    This wrapper provides two observation modes for Chronos embeddings:

    - ``"window"`` mode: Expects the environment to emit full rolling windows
      with shape ``(lookback, n_features)``. Each window is embedded
      immediately and the observation space contains the resulting embeddings.
    - ``"stream"`` mode: Collects single-step observations from the environment
      and builds rolling windows incrementally. After ``min_history`` steps, the
      wrapper begins emitting real Chronos embeddings. Before that, it returns
      warmup placeholder values.

    The wrapper automatically infers the mode from the observation space if
    ``mode="auto"`` is specified. Stream mode supports optional window expansion
    where the full retained history is embedded instead of just the last
    ``lookback`` observations, enabling dynamic effective window sizes.

    Args:
        env: Wrapped environment to provide observations.
        lookback: Length of the rolling window used by the Chronos model. Must
            be greater than 0.
        feature_names: Optional names for the per-timestep feature axis. For
            window observations this names the second axis of
            ``(lookback, n_features)``. For stream observations this names the
            flattened single-step observation. Used to document expected
            feature ordering.
        selected_columns: Optional subset of ``feature_names`` to embed by name.
            Mutually exclusive with ``selected_indices``.
        selected_indices: Optional subset of per-timestep feature positions to
            embed by index. Mutually exclusive with ``selected_columns``.
        min_history: Minimum number of collected stream observations required
            before emitting a real Chronos embedding. Defaults to ``lookback``.
            Window observations are embedded immediately because each
            observation already contains a full Chronos window. Must be greater
            than 0.
        warmup_value: Placeholder value returned for stream observations until
            ``min_history`` has been collected. Default: ``0.0``.
        mode: Observation mode to use. Valid values: ``"window"`` for envs that
            emit full Chronos windows, ``"stream"`` for envs that emit one
            timestep, or ``"auto"`` to automatically infer from the observation
            space. Default: ``"auto"``.
        expanding_window: If ``True`` in stream mode, embed the full retained
            history after warmup instead of the last ``lookback`` observations.
            This enables the effective window to grow during warmup and adapt
            after. Default: ``False``.
        max_history: Optional cap for retained stream history. When set and
            ``expanding_window=True``, keeps only the most recent ``max_history``
            embeddings. Must be at least ``min_history`` if both are specified.
        model_name: Hugging Face or Chronos model identifier to load. See
            ``ChronosEmbedder`` for supported model names. Default:
            ``"amazon/chronos-2"``.
        pooling: How token-level Chronos embeddings are pooled into one vector
            per input window. Valid values: ``"mean"`` (average token embeddings)
            or ``"last"`` (keep last token embedding). Default: ``"mean"``.
        device_map: Target device for the Chronos model, for example
            ``"auto"``, ``"cpu"``, or ``"cuda"``. Default: ``"auto"``.
        dtype: Torch dtype used when loading the Chronos pipeline. Default:
            ``torch.float32``.
        cache_size: Optional LRU cache size for repeated online windows. Improves
            performance when the same windows are encountered multiple times.
            Default: ``16384``.

    Raises:
        ValueError: If ``lookback`` <= 0, ``min_history`` <= 0, ``max_history``
            is invalid, or other parameter constraints are violated.

    Example::

        # Stream mode: collect single-step observations into rolling windows
        wrapper = WalkForwardChronosWrapper(
            env,
            lookback=32,
            mode="stream",
            min_history=32,
        )
        obs, info = wrapper.reset()
        obs.shape  # (embedding_dim,)

        # Window mode: embed pre-formed rolling windows
        wrapper = WalkForwardChronosWrapper(
            env,
            lookback=32,
            mode="window",
        )
        obs, info = wrapper.reset()
        obs.shape  # (embedding_dim,)
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
        mode: ObservationMode = "auto",
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
            mode=mode,
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
        self._observation_adapter = ChronosObservationAdapter(
            env.observation_space,
            lookback=self.lookback,
            observation_mode=mode,
            feature_names=feature_names,
        )
        self.mode = self._observation_adapter.mode
        self.n_features = self._observation_adapter.n_features
        self.feature_names = self._observation_adapter.feature_names

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

    def _append_observation(self, observation: Any) -> None:
        flattened = self._observation_adapter.flatten_stream_observation(observation)
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
        if self.mode == "window":
            raise RuntimeError("_embedded_observation is only used in stream mode.")

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

    def _embed_window_observation(self, observation: Any) -> np.ndarray:
        window = self._observation_adapter.window_from_observation(observation)
        embedding = self.embedder.embed_windows(
            window,
            lookback=self.lookback,
            n_features=self.n_features,
            feature_names=self.feature_names,
            as_tensor=False,
        )
        return embedding[0].astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if self.mode == "window":
            return self._embed_window_observation(observation), info

        self._history = []
        self._append_observation(observation)
        return self._embedded_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.mode == "window":
            return (
                self._embed_window_observation(observation),
                reward,
                terminated,
                truncated,
                info,
            )

        self._append_observation(observation)
        return self._embedded_observation(), reward, terminated, truncated, info


__all__ = ["WalkForwardChronosWrapper"]

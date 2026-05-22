from __future__ import annotations

from typing import Any, Literal, Sequence

import gymnasium as gym
import numpy as np
import torch

from crosslearn._devices import resolve_device
from crosslearn.envs._chronos_observation import ChronosObservationAdapter, ObservationMode
from crosslearn.extractors.chronos import (
    ChronosEmbedder,
    _resolve_dataframe_feature_names,
)
from crosslearn.extractors.pca import (
    _PCAFitState,
    _fit_pca,
    _project_rows,
    _resolve_n_components,
    _select_n_components,
    _to_numpy_float32,
    _validate_compute_dtype,
    _validate_requested_n_components,
    _validate_solver,
)


class WalkForwardChronosPCAWrapper(
    gym.Wrapper[np.ndarray, Any, np.ndarray, Any],
    gym.utils.RecordConstructorArgs,
):
    """Online Chronos embeddings with walk-forward adaptive PCA dimensionality reduction.

    This wrapper combines ``WalkForwardChronosWrapper`` with adaptive principal
    component analysis (PCA) to reduce Chronos embedding dimensionality during
    environment execution. It supports two initialization modes:

    - **Dataframe mode** (``fit_source="dataframe"``): Fits initial PCA from
      historical Chronos embeddings computed offline on the dataframe. Requires
      a pandas-like dataframe, ``frame_bound``, and ``mode="window"``. After
      initialization, PCA adapts online as new observations arrive.
    - **Online mode** (``fit_source="online"``): Fits PCA incrementally from
      live Chronos embeddings during environment execution. Requires explicit
      ``n_components`` (since observation space must be fixed before warmup
      completes). No dataframe required.

    In both modes, after collecting ``min_history`` embeddings, the wrapper
    continuously re-fits PCA and projects new embeddings onto the principal
    subspace. The ``expanding_window`` option allows the effective history to
    grow during warmup, and ``max_history`` caps total retained embeddings.

    Args:
        env: Wrapped environment to provide observations.
        lookback: Length of the rolling window used by Chronos. Must be greater
            than 0.
        min_history: Minimum number of Chronos embeddings required before PCA is
            fitted and observations are emitted. Must be at least 2 for PCA to
            be computable. If ``max_history`` is set with ``expanding_window=True``,
            must not exceed ``max_history``.
        warmup_value: Placeholder value returned for projected embeddings during
            the warmup phase (when fewer than ``min_history`` embeddings have
            been collected). Default: ``0.0``.
        mode: Observation mode passed to the underlying ``ChronosObservationAdapter``.
            Valid values: ``"window"``, ``"stream"``, or ``"auto"`` to infer
            from observation space. Default: ``"auto"``.
        feature_names: Optional names for the per-timestep feature axis used when
            reading observations. For window observations this names the second
            axis of ``(lookback, n_features)``.
        df: Optional pandas-like dataframe used for dataframe-mode PCA fitting.
            When provided with ``mode="window"`` and ``frame_bound``, enables
            offline pre-fit initialization. If not provided, the wrapper looks
            for ``env.df`` attribute.
        frame_bound: Two-element tuple ``(start_idx, end_idx)`` defining the
            environment's agent-visible time range. Required for dataframe mode.
            If not provided, the wrapper looks for ``env.frame_bound`` attribute.
        history_frame_bound: Optional tuple ``(start_idx, end_idx)`` for the
            warmup history slice used to fit initial PCA in dataframe mode.
            If omitted, automatically set to match the warmup region.
        selected_columns: Optional subset of ``feature_names`` to embed by name
            in the Chronos embedder. Mutually exclusive with ``selected_indices``.
        selected_indices: Optional subset of feature positions to embed by index
            in the Chronos embedder. Mutually exclusive with ``selected_columns``.
        expanding_window: If ``True``, retain the full embeddings history after
            warmup instead of capping at ``min_history``. Enables dynamic
            effective history size. Default: ``True``.
        max_history: Optional cap on the maximum number of retained embeddings.
            When ``expanding_window=True``, keeps only the most recent
            ``max_history`` embeddings. Must be at least ``min_history`` if both
            are specified.
        explained_variance_threshold: Target cumulative explained variance ratio
            for automatic ``n_components`` selection in dataframe mode. For example,
            ``0.99`` selects enough components to explain 99% of variance.
            Ignored if ``n_components`` is explicitly provided. Default: ``0.99``.
        n_components: Explicit number of PCA components to use. When omitted in
            dataframe mode, automatically selected to meet
            ``explained_variance_threshold``. Required in online mode since the
            observation space must be fixed before warmup completes.
        standardize: Whether to standardize (z-score normalize) embeddings before
            PCA fitting and projection. Default: ``True``.
        solver: Solver algorithm for PCA: ``"svd"`` (default) or
            ``"covariance_eigh"``. ``"svd"`` is more stable for online fitting
            where ``n_components`` may be close to ``min_history``.
            Default: ``"svd"``.
        compute_dtype: Torch dtype used internally for PCA computations. Using
            higher precision (e.g., ``torch.float64``) improves numerical
            stability. Default: ``torch.float64``.
        model_name: Hugging Face or Chronos model identifier for the underlying
            embedder. Default: ``"amazon/chronos-2"``.
        pooling: Token pooling mode for the Chronos embedder. Valid values:
            ``"mean"`` or ``"last"``. Default: ``"mean"``.
        device_map: Target device for the Chronos embedder (e.g., ``"auto"``,
            ``"cpu"``, ``"cuda"``). Default: ``"auto"``.
        pca_device: Target device for PCA computations. If not provided, defaults
            to ``device_map``. Specify separately if CPU PCA is preferred while
            embeddings run on GPU. Default: ``None`` (uses ``device_map``).
        dtype: Torch dtype used when loading the Chronos pipeline. Default:
            ``torch.float32``.
        cache_size: Optional LRU cache size for the Chronos embedder to cache
            repeated windows. Default: ``16384``.

    Raises:
        ValueError: If constraints are violated (e.g., ``lookback`` <= 0,
            ``min_history`` < 2, ``n_components`` exceeds embedding dimension
            or ``min_history`` in online mode, dataframe mode without required
            parameters, etc.).

    Example::

        # Dataframe mode: pre-fit PCA from historical data
        wrapper = WalkForwardChronosPCAWrapper(
            env,
            lookback=32,
            min_history=32,
            df=historical_df,
            frame_bound=(1000, 2000),
            explained_variance_threshold=0.99,
        )
        obs, info = wrapper.reset()
        obs.shape  # (n_components,) where n_components meets 0.99 variance

        # Online mode: fit PCA from live embeddings
        wrapper = WalkForwardChronosPCAWrapper(
            env,
            lookback=32,
            min_history=64,
            n_components=16,  # Required in online mode
            mode="stream",
        )
        obs, info = wrapper.reset()
        obs.shape  # (16,)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        lookback: int,
        min_history: int,
        warmup_value: float = 0.0,
        mode: ObservationMode = "auto",
        feature_names: Sequence[str] | None = None,
        df: Any | None = None,
        frame_bound: Sequence[int] | None = None,
        history_frame_bound: Sequence[int] | None = None,
        selected_columns: Sequence[str] | None = None,
        selected_indices: Sequence[int] | None = None,
        expanding_window: bool = True,
        max_history: int | None = None,
        explained_variance_threshold: float = 0.99,
        n_components: int | None = None,
        standardize: bool = True,
        solver: Literal["svd", "covariance_eigh"] = "svd",
        compute_dtype: torch.dtype = torch.float64,
        model_name: str = "amazon/chronos-2",
        pooling: str = "mean",
        device_map: str | torch.device = "auto",
        pca_device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        cache_size: int | None = 16_384,
    ) -> None:
        gym.utils.RecordConstructorArgs.__init__(
            self,
            lookback=lookback,
            min_history=min_history,
            warmup_value=warmup_value,
            mode=mode,
            feature_names=list(feature_names) if feature_names is not None else None,
            frame_bound=tuple(frame_bound) if frame_bound is not None else None,
            history_frame_bound=(
                tuple(history_frame_bound)
                if history_frame_bound is not None
                else None
            ),
            selected_columns=(
                list(selected_columns) if selected_columns is not None else None
            ),
            selected_indices=(
                [int(index) for index in selected_indices]
                if selected_indices is not None
                else None
            ),
            expanding_window=expanding_window,
            max_history=max_history,
            explained_variance_threshold=explained_variance_threshold,
            n_components=n_components,
            standardize=standardize,
            solver=solver,
            compute_dtype=compute_dtype,
            model_name=model_name,
            pooling=pooling,
            device_map=device_map,
            pca_device=str(pca_device) if pca_device is not None else None,
            dtype=dtype,
            cache_size=cache_size,
        )
        super().__init__(env)

        self.lookback = int(lookback)
        if self.lookback <= 0:
            raise ValueError("lookback must be greater than 0.")

        self.min_history = int(min_history)
        if self.min_history < 2:
            raise ValueError("min_history must be at least 2 for PCA.")

        self.warmup_value = float(warmup_value)
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

        self.standardize = bool(standardize)
        self.solver = _validate_solver(solver)
        self.compute_dtype = _validate_compute_dtype(compute_dtype)
        self.explained_variance_threshold = float(explained_variance_threshold)
        self.requested_n_components = _validate_requested_n_components(n_components)
        self.pca_device = resolve_device(
            pca_device if pca_device is not None else device_map
        )

        source_df = df if df is not None else getattr(env, "df", None)
        preliminary_adapter = ChronosObservationAdapter(
            env.observation_space,
            lookback=self.lookback,
            observation_mode=mode,
            feature_names=feature_names,
        )
        self.mode = preliminary_adapter.mode
        self.fit_source = self._resolve_fit_source(
            source_df=source_df,
            frame_bound=frame_bound,
            resolved_mode=self.mode,
        )

        if self.fit_source == "dataframe":
            self._init_dataframe_mode(
                env=env,
                source_df=source_df,
                frame_bound=frame_bound,
                history_frame_bound=history_frame_bound,
                feature_names=feature_names,
                mode=mode,
                selected_columns=selected_columns,
                selected_indices=selected_indices,
                model_name=model_name,
                pooling=pooling,
                device_map=device_map,
                dtype=dtype,
                cache_size=cache_size,
            )
        else:
            self._init_online_mode(
                env=env,
                feature_names=feature_names,
                mode=mode,
                selected_columns=selected_columns,
                selected_indices=selected_indices,
                model_name=model_name,
                pooling=pooling,
                device_map=device_map,
                dtype=dtype,
                cache_size=cache_size,
            )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_components,),
            dtype=np.float32,
        )

    def _resolve_fit_source(
        self,
        *,
        source_df: Any | None,
        frame_bound: Sequence[int] | None,
        resolved_mode: Literal["window", "stream"],
    ) -> Literal["dataframe", "online"]:
        env_frame_bound = getattr(self.env, "frame_bound", None)
        has_frame_bound = frame_bound is not None or env_frame_bound is not None
        if resolved_mode == "window" and source_df is not None and has_frame_bound:
            return "dataframe"
        return "online"

    def _make_embedder(
        self,
        *,
        model_name: str,
        pooling: str,
        selected_columns: Sequence[str] | None,
        selected_indices: Sequence[int] | None,
        device_map: str | torch.device,
        dtype: torch.dtype,
        cache_size: int | None,
    ) -> ChronosEmbedder:
        return ChronosEmbedder(
            model_name=model_name,
            pooling=pooling,
            feature_names=self.feature_names,
            selected_columns=selected_columns,
            selected_indices=selected_indices,
            device_map=device_map,
            dtype=dtype,
            cache_size=cache_size,
        )

    def _init_dataframe_mode(
        self,
        *,
        env: gym.Env,
        source_df: Any | None,
        frame_bound: Sequence[int] | None,
        history_frame_bound: Sequence[int] | None,
        feature_names: Sequence[str] | None,
        mode: ObservationMode,
        selected_columns: Sequence[str] | None,
        selected_indices: Sequence[int] | None,
        model_name: str,
        pooling: str,
        device_map: str | torch.device,
        dtype: torch.dtype,
        cache_size: int | None,
    ) -> None:
        if source_df is None:
            raise ValueError(
                "Dataframe-backed Chronos + PCA mode requires df=... or env.df."
            )

        self.df = source_df.reset_index(drop=True).copy()
        self.feature_names = _resolve_dataframe_feature_names(
            self.df,
            feature_names=feature_names,
            context="Chronos + PCA wrapper",
        )
        self._observation_adapter = ChronosObservationAdapter(
            env.observation_space,
            lookback=self.lookback,
            observation_mode=mode,
            feature_names=self.feature_names,
        )
        self.mode = self._observation_adapter.mode
        self.n_features = self._observation_adapter.n_features

        self.agent_frame_bound = (
            tuple(int(bound) for bound in frame_bound)
            if frame_bound is not None
            else tuple(int(bound) for bound in getattr(env, "frame_bound", ()))
        )
        if len(self.agent_frame_bound) != 2:
            raise ValueError(
                "Dataframe-backed Chronos + PCA mode requires frame_bound=(start, end) "
                "or an env.frame_bound attribute."
            )

        if self.agent_frame_bound[0] < self.lookback + self.min_history:
            raise ValueError(
                "frame_bound[0] must be at least lookback + min_history for the "
                "Chronos + PCA wrapper."
            )
        if self.agent_frame_bound[1] <= self.agent_frame_bound[0]:
            raise ValueError(
                "frame_bound[1] must be greater than frame_bound[0] so the wrapper "
                "has at least one post-warmup observation to return."
            )
        if self.agent_frame_bound[1] > len(self.df):
            raise ValueError(
                f"frame_bound[1] must be <= len(df)={len(self.df)}, got "
                f"{self.agent_frame_bound[1]}."
            )

        if history_frame_bound is not None:
            resolved_history_frame_bound = tuple(int(bound) for bound in history_frame_bound)
            expected_history_start = self.agent_frame_bound[0] - self.min_history
            if resolved_history_frame_bound != (
                expected_history_start,
                self.agent_frame_bound[1],
            ):
                raise ValueError(
                    "history_frame_bound must equal "
                    f"({expected_history_start}, {self.agent_frame_bound[1]}) "
                    "for the Chronos + PCA wrapper."
                )

        self.embedder = self._make_embedder(
            model_name=model_name,
            pooling=pooling,
            selected_columns=selected_columns,
            selected_indices=selected_indices,
            device_map=device_map,
            dtype=dtype,
            cache_size=cache_size,
        )

        warmup_windows = np.stack(
            [
                self._build_dataframe_window(end_index)
                for end_index in range(
                    self.agent_frame_bound[0] - self.min_history,
                    self.agent_frame_bound[0],
                )
            ],
            axis=0,
        )
        warmup_embeddings = self.embedder.embed_windows(
            warmup_windows,
            lookback=self.lookback,
            n_features=self.n_features,
            feature_names=self.feature_names,
            as_tensor=True,
            output_device=self.pca_device,
        )
        self._warmup_embeddings = warmup_embeddings.detach().to(dtype=torch.float32)

        initial_state = _fit_pca(
            self._warmup_embeddings,
            standardize=self.standardize,
            solver=self.solver,
            compute_dtype=self.compute_dtype,
        )
        self.threshold_n_components = _select_n_components(
            initial_state.explained_variance_ratio,
            self.explained_variance_threshold,
        )
        self.n_components = _resolve_n_components(
            requested_n_components=self.requested_n_components,
            threshold_n_components=self.threshold_n_components,
            explained_variance_threshold=self.explained_variance_threshold,
        )
        self._initial_fit_state = _PCAFitState(
            mean=initial_state.mean,
            scale=initial_state.scale,
            components=initial_state.components[: self.n_components].clone(),
            explained_variance_ratio=initial_state.explained_variance_ratio.clone(),
        )
        self._initial_reference_components = self._initial_fit_state.components.clone()

        self._history_embeddings: list[torch.Tensor] = []
        self._current_embedding: torch.Tensor | None = None
        self._reference_components: torch.Tensor | None = None

    def _init_online_mode(
        self,
        *,
        env: gym.Env,
        feature_names: Sequence[str] | None,
        mode: ObservationMode,
        selected_columns: Sequence[str] | None,
        selected_indices: Sequence[int] | None,
        model_name: str,
        pooling: str,
        device_map: str | torch.device,
        dtype: torch.dtype,
        cache_size: int | None,
    ) -> None:
        if self.requested_n_components is None:
            raise ValueError(
                "Online Chronos + PCA mode requires explicit n_components because "
                "observation_space must be fixed before live PCA warmup completes."
            )

        self._observation_adapter = ChronosObservationAdapter(
            env.observation_space,
            lookback=self.lookback,
            observation_mode=mode,
            feature_names=feature_names,
        )
        self.mode = self._observation_adapter.mode
        self.n_features = self._observation_adapter.n_features
        self.feature_names = self._observation_adapter.feature_names
        self.n_components = self.requested_n_components
        self.threshold_n_components = None

        self.embedder = self._make_embedder(
            model_name=model_name,
            pooling=pooling,
            selected_columns=selected_columns,
            selected_indices=selected_indices,
            device_map=device_map,
            dtype=dtype,
            cache_size=cache_size,
        )

        example_embedding = self.embedder.embed_windows(
            np.zeros((self.lookback, self.n_features), dtype=np.float32),
            lookback=self.lookback,
            n_features=self.n_features,
            feature_names=self.feature_names,
            as_tensor=True,
            output_device=self.pca_device,
        )
        self.embedder.clear_cache()
        embedding_dim = int(example_embedding.shape[-1])
        if self.n_components > embedding_dim:
            raise ValueError(
                f"n_components={self.n_components} exceeds Chronos embedding "
                f"width {embedding_dim}."
            )
        if self.solver == "svd" and self.n_components > self.min_history:
            raise ValueError(
                "n_components must be <= min_history when online Chronos + PCA "
                "and solver='svd'."
            )

        self._history_embeddings: list[torch.Tensor] = []
        self._reference_components: torch.Tensor | None = None

    def _build_dataframe_window(self, end_index: int) -> np.ndarray:
        if end_index < self.lookback:
            raise ValueError(
                f"end_index={end_index} does not have enough history for "
                f"lookback={self.lookback}."
            )
        if end_index > len(self.df):
            raise ValueError(f"end_index={end_index} exceeds len(df)={len(self.df)}.")
        return self.df.iloc[end_index - self.lookback : end_index][
            self.feature_names
        ].to_numpy(dtype=np.float32, copy=True)

    def _window_from_observation(self, observation: Any) -> np.ndarray | None:
        if self.mode == "window":
            return self._observation_adapter.window_from_observation(observation)

        step = self._observation_adapter.flatten_stream_observation(observation)
        self._stream_history.append(step.copy())
        retained_history = max(self.lookback, self.min_history)
        if len(self._stream_history) > retained_history:
            self._stream_history = self._stream_history[-retained_history:]
        if len(self._stream_history) < self.lookback:
            return None
        return np.stack(self._stream_history[-self.lookback :], axis=0)

    def _embed_window(self, window: np.ndarray) -> torch.Tensor:
        embedded = self.embedder.embed_windows(
            window,
            lookback=self.lookback,
            n_features=self.n_features,
            feature_names=self.feature_names,
            as_tensor=True,
            output_device=self.pca_device,
        )
        return embedded[0].detach().to(dtype=torch.float32)

    def _project_embedding(
        self,
        embedding: torch.Tensor,
        fit_state: _PCAFitState,
    ) -> np.ndarray:
        return _to_numpy_float32(_project_rows(embedding.reshape(1, -1), fit_state)[0])

    def _trim_pca_history(self) -> None:
        if self.expanding_window:
            if self.max_history is not None and len(self._history_embeddings) > self.max_history:
                self._history_embeddings = self._history_embeddings[-self.max_history :]
        elif len(self._history_embeddings) > self.min_history:
            self._history_embeddings = self._history_embeddings[-self.min_history :]

    def _online_observation_from_embedding(self, embedding: torch.Tensor | None) -> np.ndarray:
        if embedding is None:
            return np.full((self.n_components,), self.warmup_value, dtype=np.float32)

        if len(self._history_embeddings) < self.min_history:
            self._history_embeddings.append(embedding.clone())
            self._trim_pca_history()
            return np.full((self.n_components,), self.warmup_value, dtype=np.float32)

        fit_state = _fit_pca(
            torch.stack(self._history_embeddings, dim=0),
            standardize=self.standardize,
            n_components=self.n_components,
            reference_components=self._reference_components,
            solver=self.solver,
            compute_dtype=self.compute_dtype,
        )
        self._reference_components = fit_state.components.clone()
        projected = self._project_embedding(embedding, fit_state)
        self._history_embeddings.append(embedding.clone())
        self._trim_pca_history()
        return projected

    def reset(self, *, seed: int | None = None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)

        if self.fit_source == "dataframe":
            self._history_embeddings = [row.clone() for row in self._warmup_embeddings]
            self._reference_components = self._initial_reference_components.clone()
            current_embedding = self._embed_window(
                self._observation_adapter.window_from_observation(observation)
            )
            projected = self._project_embedding(current_embedding, self._initial_fit_state)
            self._current_embedding = current_embedding
            return projected, info

        self._stream_history: list[np.ndarray] = []
        self._history_embeddings = []
        self._reference_components = None
        window = self._window_from_observation(observation)
        embedding = None if window is None else self._embed_window(window)
        return self._online_observation_from_embedding(embedding), info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.fit_source == "dataframe":
            if self._current_embedding is None or self._reference_components is None:
                raise RuntimeError(
                    "WalkForwardChronosPCAWrapper.step() called before reset()."
                )

            self._history_embeddings.append(self._current_embedding.clone())
            self._trim_pca_history()
            fit_state = _fit_pca(
                torch.stack(self._history_embeddings, dim=0),
                standardize=self.standardize,
                n_components=self.n_components,
                reference_components=self._reference_components,
                solver=self.solver,
                compute_dtype=self.compute_dtype,
            )
            self._reference_components = fit_state.components.clone()
            window = self._observation_adapter.window_from_observation(observation)
            self._current_embedding = self._embed_window(window)
            projected = self._project_embedding(self._current_embedding, fit_state)
            return projected, reward, terminated, truncated, info

        window = self._window_from_observation(observation)
        embedding = None if window is None else self._embed_window(window)
        projected = self._online_observation_from_embedding(embedding)
        return projected, reward, terminated, truncated, info


__all__ = ["WalkForwardChronosPCAWrapper"]

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from crosslearn.extractors.chronos import ChronosEmbedder
from crosslearn.extractors.pca import _PCAFitState, _fit_pca, _project_rows, _select_n_components


class WalkForwardChronosPCAWrapper(
    gym.Wrapper[np.ndarray, int, np.ndarray, int],
    gym.utils.RecordConstructorArgs,
):
    """Online Chronos + walk-forward PCA wrapper for sequential dataframe envs.

    By design, the first ``lookback + warmup`` environment observations are
    skipped. The first agent-visible observation is the next one after that
    skipped prefix, so the wrapper requires enough data for the skipped history
    plus at least one projected observation.

    The wrapper does not precompute the full episode representation. Instead it:

    1. builds the initial warmup embedding history from rows before the first
       agent-visible observation
    2. fits PCA on that warmup history once to determine the fixed component count
    3. embeds the current raw observation at reset and projects it with the
       warmup PCA fit
    4. on each step, appends the previously projected embedding to history,
       refits PCA on that expanding history, embeds the newly returned raw
       observation, and projects only that new embedding

    This keeps the Chronos and PCA computation online at the environment layer,
    while still preserving a stable observation-space width for the policy.

    The wrapped environment is expected to emit raw rolling-window
    observations compatible with ``ChronosEmbedder.embed_windows(...)`` for a
    single window, typically ``(lookback, len(feature_columns))``.

    Args:
        env: Sequential wrapped environment. It should be deterministic with
            respect to the underlying dataframe history and should emit one raw
            rolling-window observation per step.
        lookback: Number of rows per raw observation window.
        warmup: Number of initial embedded observations used to fit the first
            PCA state and determine the fixed component count.
        feature_columns: Raw dataframe columns used to reconstruct historical
            windows for the Chronos warmup history.
        df: Optional source dataframe. When omitted, ``env.df`` is used.
        frame_bound: Optional two-element agent-visible span. When omitted,
            ``env.frame_bound`` is used. The wrapper requires
            ``frame_bound[0] >= lookback + warmup`` and
            ``frame_bound[1] > frame_bound[0]``.
        history_frame_bound: Optional consistency-checked alias for the implied
            history slice ``(frame_bound[0] - warmup, frame_bound[1])``. It is
            not an independent tuning knob.
        selected_columns: Optional subset of ``feature_columns`` to embed by
            name before Chronos is called.
        selected_indices: Optional subset of ``feature_columns`` to embed by
            index before Chronos is called.
        explained_variance_threshold: Cumulative explained-variance threshold
            used on the initial warmup PCA fit to choose the fixed component
            count.
        standardize: If ``True``, recompute mean/std walk-forward alongside the
            PCA loadings. If ``False``, PCA is still centered but not
            variance-scaled.
        progress_bar: Reserved for API parity with the offline helpers. The
            current online wrapper does not display a progress bar.
        model_name: Chronos model identifier. Default: ``"amazon/chronos-2"``.
        pooling: Token pooling mode forwarded to ``ChronosEmbedder``.
        device_map: Target device for the Chronos model.
        dtype: Torch dtype used when loading Chronos.

    Example::

        env = WalkForwardChronosPCAWrapper(
            env,
            lookback=32,
            warmup=500,
            feature_columns=["open", "high", "low", "close", "volume"],
            frame_bound=(532, len(df)),
        )
        obs, info = env.reset()
        obs.shape  # (n_components,)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        lookback: int,
        warmup: int,
        feature_columns: Sequence[str],
        df: Any | None = None,
        frame_bound: Sequence[int] | None = None,
        history_frame_bound: Sequence[int] | None = None,
        selected_columns: Sequence[str] | None = None,
        selected_indices: Sequence[int] | None = None,
        explained_variance_threshold: float = 0.99,
        standardize: bool = True,
        progress_bar: bool = False,
        model_name: str = "amazon/chronos-2",
        pooling: str = "mean",
        device_map: str = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        gym.utils.RecordConstructorArgs.__init__(
            self,
            lookback=lookback,
            warmup=warmup,
            feature_columns=list(feature_columns),
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
            explained_variance_threshold=explained_variance_threshold,
            standardize=standardize,
            progress_bar=progress_bar,
            model_name=model_name,
            pooling=pooling,
            device_map=device_map,
            dtype=dtype,
        )
        super().__init__(env)

        source_df = df if df is not None else getattr(env, "df", None)
        if source_df is None:
            raise ValueError(
                "WalkForwardChronosPCAWrapper requires a dataframe via df=... or env.df."
            )

        self.df = source_df.reset_index(drop=True).copy()
        self.lookback = int(lookback)
        self.warmup = int(warmup)
        if self.warmup < 2:
            raise ValueError("warmup must be at least 2 for PCA.")
        self.feature_columns = [str(column) for column in feature_columns]
        if not self.feature_columns:
            raise ValueError("feature_columns must contain at least one column name.")
        missing_columns = [
            column for column in self.feature_columns if column not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing dataframe columns for Chronos + PCA wrapper: {missing_columns}"
            )

        self.agent_frame_bound = (
            tuple(int(bound) for bound in frame_bound)
            if frame_bound is not None
            else tuple(int(bound) for bound in getattr(env, "frame_bound", ()))
        )
        if len(self.agent_frame_bound) != 2:
            raise ValueError(
                "WalkForwardChronosPCAWrapper requires frame_bound=(start, end) "
                "or an env.frame_bound attribute."
            )

        if self.agent_frame_bound[0] < self.lookback + self.warmup:
            raise ValueError(
                "frame_bound[0] must be at least lookback + warmup for the online "
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
            expected_history_start = self.agent_frame_bound[0] - self.warmup
            if resolved_history_frame_bound != (
                expected_history_start,
                self.agent_frame_bound[1],
            ):
                raise ValueError(
                    "history_frame_bound must equal "
                    f"({expected_history_start}, {self.agent_frame_bound[1]}) "
                    "for the online Chronos + PCA wrapper."
                )

        self.standardize = bool(standardize)
        self.explained_variance_threshold = float(explained_variance_threshold)
        self.progress_bar = bool(progress_bar)

        self.embedder = ChronosEmbedder(
            model_name=model_name,
            pooling=pooling,
            feature_names=self.feature_columns,
            selected_columns=selected_columns,
            selected_indices=selected_indices,
            device_map=device_map,
            dtype=dtype,
        )

        warmup_windows = np.stack(
            [
                self._build_window(end_index)
                for end_index in range(
                    self.agent_frame_bound[0] - self.warmup,
                    self.agent_frame_bound[0],
                )
            ],
            axis=0,
        )
        warmup_embeddings = self.embedder.embed_windows(
            warmup_windows,
            lookback=self.lookback,
            n_features=len(self.feature_columns),
            feature_names=self.feature_columns,
            as_tensor=False,
        )
        self._warmup_embeddings = np.asarray(warmup_embeddings, dtype=np.float32)

        initial_state = _fit_pca(
            self._warmup_embeddings,
            standardize=self.standardize,
        )
        self.n_components = _select_n_components(
            initial_state.explained_variance_ratio,
            self.explained_variance_threshold,
        )
        self._initial_fit_state = _PCAFitState(
            mean=initial_state.mean,
            scale=initial_state.scale,
            components=initial_state.components[: self.n_components].copy(),
            explained_variance_ratio=initial_state.explained_variance_ratio.copy(),
        )
        self._initial_reference_components = self._initial_fit_state.components.copy()

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_components,),
            dtype=np.float32,
        )

        self._history_embeddings: list[np.ndarray] = []
        self._current_embedding: np.ndarray | None = None
        self._reference_components: np.ndarray | None = None

    def _build_window(self, end_index: int) -> np.ndarray:
        """Return one historical raw window ending at ``end_index``.

        Args:
            end_index: Exclusive dataframe end index for the window.

        Returns:
            A ``float32`` array of shape ``(lookback, len(feature_columns))``.
        """
        if end_index < self.lookback:
            raise ValueError(
                f"end_index={end_index} does not have enough history for lookback={self.lookback}."
            )
        if end_index > len(self.df):
            raise ValueError(f"end_index={end_index} exceeds len(df)={len(self.df)}.")
        return self.df.iloc[end_index - self.lookback : end_index][
            self.feature_columns
        ].to_numpy(dtype=np.float32, copy=True)

    def _embed_single_observation(self, observation: np.ndarray) -> np.ndarray:
        """Embed one wrapped-env observation into a single Chronos vector.

        Args:
            observation: Raw observation emitted by the wrapped env for one
                timestep.

        Returns:
            A ``float32`` embedding vector of shape ``(embedding_dim,)``.
        """
        embedded = self.embedder.embed_windows(
            observation,
            lookback=self.lookback,
            n_features=len(self.feature_columns),
            feature_names=self.feature_columns,
            as_tensor=False,
        )
        return np.asarray(embedded[0], dtype=np.float32)

    def _project_current_embedding(self, fit_state: _PCAFitState) -> np.ndarray:
        """Project the current Chronos embedding with a supplied PCA state.

        Args:
            fit_state: Walk-forward PCA state containing the mean, optional
                scale, and PCA loadings to apply.

        Returns:
            A ``float32`` PCA observation of shape ``(n_components,)``.
        """
        if self._current_embedding is None:
            raise RuntimeError("No current embedding is available to project.")
        return _project_rows(self._current_embedding.reshape(1, -1), fit_state)[0]

    def reset(self, *, seed: int | None = None, options=None):
        """Reset the wrapped env and emit the first projected PCA observation.

        The raw observation returned by ``env.reset(...)`` is embedded and
        projected with PCA fit on the warmup embedding history. The raw
        observation itself is not returned to the agent.

        Args:
            seed: Optional reset seed forwarded to the wrapped env.
            options: Optional reset options forwarded to the wrapped env.

        Returns:
            A tuple ``(observation, info)`` where ``observation`` is a
            ``float32`` vector of shape ``(n_components,)`` and ``info`` is the
            wrapped env's reset info dict.
        """
        observation, info = self.env.reset(seed=seed, options=options)
        self._history_embeddings = [row.copy() for row in self._warmup_embeddings]
        self._reference_components = self._initial_reference_components.copy()
        self._current_embedding = self._embed_single_observation(observation)
        projected = self._project_current_embedding(self._initial_fit_state)
        return projected, info

    def step(self, action: int):
        """Advance the env and emit the next online Chronos + PCA observation.

        Before projecting the newly returned raw observation, the wrapper
        appends the previously used Chronos embedding to history and refits PCA
        on that expanding history with the fixed component count chosen at
        initialization.

        Args:
            action: Action to forward to the wrapped env.

        Returns:
            The standard Gymnasium step tuple
            ``(observation, reward, terminated, truncated, info)`` where
            ``observation`` is the newly projected PCA vector rather than the
            wrapped env's raw rolling window.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._current_embedding is None or self._reference_components is None:
            raise RuntimeError(
                "WalkForwardChronosPCAWrapper.step() called before reset()."
            )

        self._history_embeddings.append(self._current_embedding.copy())
        fit_state = _fit_pca(
            np.stack(self._history_embeddings, axis=0),
            standardize=self.standardize,
            n_components=self.n_components,
            reference_components=self._reference_components,
        )
        self._reference_components = fit_state.components.copy()
        self._current_embedding = self._embed_single_observation(observation)
        projected = self._project_current_embedding(fit_state)
        return projected, reward, terminated, truncated, info


__all__ = ["WalkForwardChronosPCAWrapper"]

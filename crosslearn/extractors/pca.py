"""Walk-forward PCA helpers for time-series-safe dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class _PCAFitState:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray


def _make_dataframe_progress_bar(total_rows: int) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "Walk-forward PCA dataframe progress bars require tqdm.\n"
            "Install it directly, for example:\n"
            "  pip install tqdm"
        ) from exc

    return tqdm(
        total=total_rows,
        unit="row",
        desc="Walk-forward PCA",
        dynamic_ncols=True,
    )


def _as_2d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_rows, n_features).")
    if array.shape[1] == 0:
        raise ValueError("Expected at least one feature column.")
    return array


def _select_n_components(
    explained_variance_ratio: np.ndarray,
    explained_variance_threshold: float,
) -> int:
    cumulative = np.cumsum(explained_variance_ratio, dtype=np.float64)
    selected = int(
        np.searchsorted(cumulative, explained_variance_threshold, side="left") + 1
    )
    return min(selected, int(explained_variance_ratio.shape[0]))


def _fit_pca(
    history: np.ndarray,
    *,
    standardize: bool,
    n_components: int | None = None,
    reference_components: np.ndarray | None = None,
) -> _PCAFitState:
    history_f64 = history.astype(np.float64, copy=False)
    mean = history_f64.mean(axis=0)
    transformed_history = history_f64 - mean

    if standardize:
        raw_scale = history_f64.std(axis=0, ddof=0)
        scale = np.where(raw_scale > 0.0, raw_scale, 1.0)
        transformed_history = transformed_history / scale
    else:
        scale = np.ones(history.shape[1], dtype=np.float64)

    _, singular_values, vt = np.linalg.svd(transformed_history, full_matrices=False)
    variance_denom = max(int(history.shape[0]) - 1, 1)
    explained_variance = (singular_values**2) / variance_denom
    total_variance = float(explained_variance.sum())
    if total_variance > 0.0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)

    if n_components is None:
        n_components = int(vt.shape[0])
    components = vt[:n_components].copy()

    if reference_components is not None:
        limit = min(int(reference_components.shape[0]), int(components.shape[0]))
        for index in range(limit):
            if float(np.dot(components[index], reference_components[index])) < 0.0:
                components[index] *= -1.0

    return _PCAFitState(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        components=components.astype(np.float32),
        explained_variance_ratio=explained_variance_ratio.astype(np.float32),
    )


def _project_rows(values: np.ndarray, state: _PCAFitState) -> np.ndarray:
    values_f64 = values.astype(np.float64, copy=False)
    transformed = (values_f64 - state.mean.astype(np.float64)) / state.scale.astype(
        np.float64
    )
    projected = transformed @ state.components.T.astype(np.float64)
    return projected.astype(np.float32)


class WalkForwardPCATransformer:
    """Expanding-window PCA with walk-forward scaling and projections.

    ``fit(...)`` determines the fixed component count from the initial warmup
    window. ``walkforward_transform(...)`` then keeps that width fixed while
    refitting the centering, optional standardization, and PCA loadings on the
    expanding history before projecting the next row.

    This is designed for time-series-safe dimensionality reduction. When
    ``standardize=True``, both the mean/std statistics and the PCA loadings are
    recomputed from past rows only before each next-row projection.

    Args:
        warmup: Number of initial rows used to choose the fixed PCA width and to
            fit the first PCA state. Must be at least ``2``.
        explained_variance_threshold: Cumulative explained-variance threshold
            used on the initial warmup fit to choose ``n_components_``.
        standardize: If ``True``, center and divide by the walk-forward
            standard deviation before PCA. If ``False``, PCA is still centered
            but not variance-scaled.

    Example::

        transformer = WalkForwardPCATransformer(
            warmup=500,
            explained_variance_threshold=0.99,
        )
        projected = transformer.fit_transform(values)
        projected.shape  # (len(values) - 500, transformer.n_components_)
    """

    def __init__(
        self,
        *,
        warmup: int,
        explained_variance_threshold: float = 0.99,
        standardize: bool = True,
    ) -> None:
        if warmup < 2:
            raise ValueError("warmup must be at least 2 for PCA.")
        if not 0.0 < explained_variance_threshold <= 1.0:
            raise ValueError(
                "explained_variance_threshold must be in the interval (0, 1]."
            )

        self.warmup = int(warmup)
        self.explained_variance_threshold = float(explained_variance_threshold)
        self.standardize = bool(standardize)

        self.n_features_in_: int | None = None
        self.n_components_: int | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.initial_explained_variance_ratio_: np.ndarray | None = None

    def fit(self, values: Any) -> "WalkForwardPCATransformer":
        """Fit the initial warmup PCA state and choose a fixed output width.

        Only the first ``warmup`` rows are used here. This method does not
        perform the full walk-forward projection loop.

        Args:
            values: 2D array-like input of shape ``(n_rows, n_features)``.

        Returns:
            ``self`` with ``n_components_``, ``mean_``, ``scale_``, and
            ``components_`` populated from the initial warmup fit.

        Raises:
            ValueError: If fewer than ``warmup`` rows are provided or if the
                input is not a non-empty 2D matrix.
        """
        array = _as_2d_float_array(values)
        if array.shape[0] < self.warmup:
            raise ValueError(
                f"Need at least warmup={self.warmup} rows, got {array.shape[0]}."
            )

        initial_state = _fit_pca(array[: self.warmup], standardize=self.standardize)
        selected_components = _select_n_components(
            initial_state.explained_variance_ratio,
            self.explained_variance_threshold,
        )

        self.n_features_in_ = int(array.shape[1])
        self.n_components_ = selected_components
        self.mean_ = initial_state.mean
        self.scale_ = initial_state.scale
        self.components_ = initial_state.components[:selected_components].copy()
        self.initial_explained_variance_ratio_ = (
            initial_state.explained_variance_ratio.copy()
        )
        return self

    def transform(self, values: Any) -> np.ndarray:
        """Project rows with the currently stored PCA state.

        Unlike ``walkforward_transform(...)``, this method does not refit the
        PCA state as it moves through time. It simply applies the latest fitted
        mean/scale/components to every row in ``values``.

        Args:
            values: 2D array-like input of shape ``(n_rows, n_features)``.

        Returns:
            A ``float32`` array of shape ``(n_rows, n_components_)``.

        Raises:
            ValueError: If the transformer has not been fit or if the feature
                count does not match the fitted state.
        """
        if self.components_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("WalkForwardPCATransformer must be fit before transform().")

        array = _as_2d_float_array(values)
        if int(array.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"Expected {self.n_features_in_} input features, got {array.shape[1]}."
            )

        return _project_rows(
            array,
            _PCAFitState(
                mean=self.mean_,
                scale=self.scale_,
                components=self.components_,
                explained_variance_ratio=np.empty(0, dtype=np.float32),
            ),
        )

    def fit_transform(self, values: Any) -> np.ndarray:
        """Fit on the warmup window and run the full walk-forward projection.

        This is a convenience alias for ``walkforward_transform(values)``.

        Args:
            values: 2D array-like input of shape ``(n_rows, n_features)``.

        Returns:
            A ``float32`` array of shape ``(n_rows - warmup, n_components_)``
            containing only the chronologically valid next-row projections.
        """
        return self.walkforward_transform(values)

    def walkforward_transform(
        self,
        values: Any,
        *,
        progress_bar: bool = False,
    ) -> np.ndarray:
        """Project each row using PCA refit on the expanding past only.

        The method first calls ``fit(...)`` on the initial warmup window to
        determine ``n_components_``. It then iterates from row ``warmup`` to
        the end of the matrix, refitting PCA on rows ``[:end_index]`` and
        projecting only row ``end_index``. The output therefore starts at the
        first chronologically valid next-row projection.

        Args:
            values: 2D array-like input of shape ``(n_rows, n_features)`` in
                chronological order.
            progress_bar: If ``True``, show a ``tqdm`` progress bar over the
                row-by-row walk-forward loop.

        Returns:
            A ``float32`` array of shape ``(n_rows - warmup, n_components_)``.

        Raises:
            ValueError: If the input is not a valid 2D matrix or contains fewer
                than ``warmup`` rows.
        """
        array = _as_2d_float_array(values)
        self.fit(array)

        assert self.n_components_ is not None
        assert self.components_ is not None
        assert self.mean_ is not None
        assert self.scale_ is not None

        projected_rows: list[np.ndarray] = []
        reference_components = self.components_.copy()
        latest_state = _PCAFitState(
            mean=self.mean_,
            scale=self.scale_,
            components=self.components_.copy(),
            explained_variance_ratio=np.empty(0, dtype=np.float32),
        )

        total_rows = max(int(array.shape[0]) - self.warmup, 0)
        progress = _make_dataframe_progress_bar(total_rows) if progress_bar else None
        try:
            for end_index in range(self.warmup, int(array.shape[0])):
                fit_state = _fit_pca(
                    array[:end_index],
                    standardize=self.standardize,
                    n_components=self.n_components_,
                    reference_components=reference_components,
                )
                projected_rows.append(
                    _project_rows(array[end_index : end_index + 1], fit_state)[0]
                )
                reference_components = fit_state.components.copy()
                latest_state = fit_state
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        self.mean_ = latest_state.mean
        self.scale_ = latest_state.scale
        self.components_ = latest_state.components

        if not projected_rows:
            return np.empty((0, self.n_components_), dtype=np.float32)
        return np.stack(projected_rows, axis=0).astype(np.float32, copy=False)


def walkforward_pca_dataframe(
    df: Any,
    *,
    feature_columns: Sequence[str],
    warmup: int,
    explained_variance_threshold: float = 0.99,
    standardize: bool = True,
    output_prefix: str = "pca_",
    drop_feature_columns: bool = False,
    trim_warmup: bool = False,
    progress_bar: bool = False,
) -> Any:
    """Append walk-forward PCA columns to a dataframe.

    The helper keeps the original dataframe length by default. The first
    ``warmup`` rows in the new PCA columns are ``NaN`` because no
    future-safe next-row projection exists for them yet. Set
    ``trim_warmup=True`` to drop those rows and reset the index.

    This function is intended to compose cleanly with ``embed_dataframe(...)``:
    first build Chronos embeddings, then run walk-forward PCA over the
    resulting ``chronos_*`` columns.

    Args:
        df: Source pandas dataframe in chronological order.
        feature_columns: Numeric columns to reduce with walk-forward PCA.
        warmup: Number of initial rows used to determine the fixed PCA width.
        explained_variance_threshold: Initial cumulative explained-variance
            threshold used to choose ``n_components_``.
        standardize: If ``True``, recompute mean/std walk-forward before each
            PCA fit. If ``False``, only walk-forward centering is applied.
        output_prefix: Prefix for appended PCA columns.
        drop_feature_columns: If ``True``, drop the original
            ``feature_columns`` after adding the PCA columns.
        trim_warmup: If ``True``, drop the leading warmup rows and reset the
            index so the returned dataframe contains only valid projections.
        progress_bar: If ``True``, show a ``tqdm`` progress bar over the
            walk-forward PCA loop.

    Returns:
        A copy of ``df`` with appended ``{output_prefix}*`` PCA columns. The
        returned dataframe has the same number of rows as ``df`` unless
        ``trim_warmup=True``.

    Raises:
        TypeError: If ``df`` is not a pandas dataframe.
        ValueError: If ``feature_columns`` is empty or missing from ``df``.

    Example::

        embedded = embed_dataframe(
            df,
            lookback=32,
            frame_bound=(32, len(df)),
            feature_columns=["open", "high", "low", "close", "volume"],
        )
        chronos_columns = [
            column for column in embedded.columns if column.startswith("chronos_")
        ]
        reduced = walkforward_pca_dataframe(
            embedded,
            feature_columns=chronos_columns,
            warmup=500,
            explained_variance_threshold=0.99,
            trim_warmup=True,
        )
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "walkforward_pca_dataframe requires pandas.\n"
            "Install it directly, for example:\n"
            "  pip install pandas"
        ) from exc

    if not isinstance(df, pd.DataFrame):
        raise TypeError("walkforward_pca_dataframe expects a pandas.DataFrame.")

    resolved_feature_columns = [str(column) for column in feature_columns]
    if not resolved_feature_columns:
        raise ValueError("feature_columns must contain at least one column name.")

    missing = [column for column in resolved_feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing dataframe columns for walk-forward PCA: {missing}")

    values = df.loc[:, resolved_feature_columns].to_numpy(dtype=np.float32, copy=True)
    transformer = WalkForwardPCATransformer(
        warmup=warmup,
        explained_variance_threshold=explained_variance_threshold,
        standardize=standardize,
    )
    projected = transformer.walkforward_transform(values, progress_bar=progress_bar)

    if transformer.n_components_ is None:
        raise RuntimeError("WalkForwardPCATransformer failed to resolve n_components_.")

    pca_columns = [
        f"{output_prefix}{index}" for index in range(int(transformer.n_components_))
    ]
    pca_frame = pd.DataFrame(
        data=np.nan,
        index=df.index,
        columns=pca_columns,
        dtype=np.float32,
    )
    if projected.size > 0:
        pca_frame.iloc[warmup:] = projected

    result = pd.concat([df.copy(), pca_frame], axis=1)
    if drop_feature_columns:
        result = result.drop(columns=resolved_feature_columns)
    if trim_warmup:
        result = result.iloc[warmup:].reset_index(drop=True)
    return result


__all__ = ["WalkForwardPCATransformer", "walkforward_pca_dataframe"]

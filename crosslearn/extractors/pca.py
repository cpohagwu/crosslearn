"""Walk-forward PCA helpers for time-series-safe dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from crosslearn._devices import resolve_device

_PCA_BATCH_SIZE = 256


@dataclass
class _PCAFitState:
    mean: torch.Tensor
    scale: torch.Tensor
    components: torch.Tensor
    explained_variance_ratio: torch.Tensor


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


def _as_2d_float_tensor(
    values: Any,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
    else:
        tensor = torch.as_tensor(np.asarray(values, dtype=np.float32))

    if tensor.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_rows, n_features).")
    if tensor.shape[1] == 0:
        raise ValueError("Expected at least one feature column.")
    return tensor.to(device=device, dtype=dtype)


def _to_numpy_float32(values: torch.Tensor) -> np.ndarray:
    return values.detach().to(device="cpu", dtype=torch.float32).numpy()


def _make_walkforward_windows(total_rows: int, warmup: int) -> np.ndarray:
    if total_rows <= warmup:
        return np.empty((0, 2), dtype=np.int64)

    history_stops = np.arange(warmup, total_rows, dtype=np.int64)
    target_stops = history_stops + 1
    return np.stack((history_stops, target_stops), axis=1)


def _select_n_components(
    explained_variance_ratio: torch.Tensor | np.ndarray,
    explained_variance_threshold: float,
) -> int:
    if isinstance(explained_variance_ratio, torch.Tensor):
        cumulative = torch.cumsum(
            explained_variance_ratio.to(dtype=torch.float64),
            dim=0,
        )
        threshold = torch.tensor(
            explained_variance_threshold,
            dtype=cumulative.dtype,
            device=cumulative.device,
        )
        selected = int(
            torch.searchsorted(cumulative, threshold, right=False).item() + 1
        )
    else:
        cumulative = np.cumsum(explained_variance_ratio, dtype=np.float64)
        selected = int(
            np.searchsorted(cumulative, explained_variance_threshold, side="left") + 1
        )
    return min(selected, int(explained_variance_ratio.shape[0]))


def _fit_pca(
    history: torch.Tensor | np.ndarray,
    *,
    standardize: bool,
    n_components: int | None = None,
    reference_components: torch.Tensor | np.ndarray | None = None,
) -> _PCAFitState:
    if not isinstance(history, torch.Tensor):
        history = _as_2d_float_tensor(
            history,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    history_f64 = history.to(dtype=torch.float64)
    mean = history_f64.mean(dim=0)
    transformed_history = history_f64 - mean

    if standardize:
        raw_scale = torch.std(history_f64, dim=0, unbiased=False)
        scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
        transformed_history = transformed_history / scale
    else:
        scale = torch.ones(history.shape[1], dtype=torch.float64, device=history.device)

    _, singular_values, vt = torch.linalg.svd(transformed_history, full_matrices=False)
    variance_denom = max(int(history.shape[0]) - 1, 1)
    explained_variance = (singular_values**2) / variance_denom
    total_variance = float(explained_variance.sum().item())
    if total_variance > 0.0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = torch.zeros_like(explained_variance)

    if n_components is None:
        n_components = int(vt.shape[0])
    components = vt[:n_components].clone()

    if reference_components is not None:
        if not isinstance(reference_components, torch.Tensor):
            reference_components = torch.as_tensor(
                reference_components,
                device=components.device,
                dtype=components.dtype,
            )
        else:
            reference_components = reference_components.to(
                device=components.device,
                dtype=components.dtype,
            )
        limit = min(int(reference_components.shape[0]), int(components.shape[0]))
        for index in range(limit):
            if (
                float(
                    torch.dot(components[index], reference_components[index]).item()
                )
                < 0.0
            ):
                components[index] *= -1.0

    return _PCAFitState(
        mean=mean,
        scale=scale,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
    )


def _project_rows(values: torch.Tensor | np.ndarray, state: _PCAFitState) -> torch.Tensor:
    values_f32 = _as_2d_float_tensor(
        values,
        device=state.mean.device,
        dtype=torch.float32,
    )
    values_f64 = values_f32.to(dtype=torch.float64)
    transformed = (values_f64 - state.mean) / state.scale
    projected = transformed @ state.components.T
    return projected.to(dtype=torch.float32)


def _fit_pca_batch(
    values_f64: torch.Tensor,
    history_stops: torch.Tensor,
    *,
    standardize: bool,
    n_components: int,
    cumulative: torch.Tensor | None = None,
    cumulative_sq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if history_stops.ndim != 1:
        raise ValueError("history_stops must be a 1D tensor.")
    if history_stops.numel() == 0:
        raise ValueError("history_stops must contain at least one value.")

    max_history = int(history_stops[-1].item())
    base_history = values_f64[:max_history]
    history_lengths = history_stops.to(dtype=torch.float64)

    if cumulative is None:
        cumulative = torch.cumsum(values_f64, dim=0)
    history_sum = cumulative[history_stops - 1]
    mean = history_sum / history_lengths.unsqueeze(1)

    if standardize:
        if cumulative_sq is None:
            cumulative_sq = torch.cumsum(values_f64.square(), dim=0)
        history_sq_sum = cumulative_sq[history_stops - 1]
        variance = history_sq_sum / history_lengths.unsqueeze(1) - mean.square()
        variance = torch.clamp(variance, min=0.0)
        raw_scale = torch.sqrt(variance)
        scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
    else:
        scale = torch.ones_like(mean)

    positions = torch.arange(max_history, device=values_f64.device)
    mask = positions.unsqueeze(0) < history_stops.unsqueeze(1)
    transformed_history = (
        (base_history.unsqueeze(0) - mean.unsqueeze(1)) / scale.unsqueeze(1)
    ) * mask.unsqueeze(-1).to(dtype=values_f64.dtype)

    _, singular_values, vt = torch.linalg.svd(transformed_history, full_matrices=False)
    explained_variance = singular_values.square()
    total_variance = explained_variance.sum(dim=1, keepdim=True)
    explained_variance_ratio = torch.where(
        total_variance > 0.0,
        explained_variance / total_variance,
        torch.zeros_like(explained_variance),
    )
    components = vt[:, :n_components].clone()

    return mean, scale, components, explained_variance_ratio


def _project_rows_batch(
    values: torch.Tensor | np.ndarray,
    mean: torch.Tensor,
    scale: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    values_f32 = _as_2d_float_tensor(
        values,
        device=mean.device,
        dtype=torch.float32,
    )
    values_f64 = values_f32.to(dtype=torch.float64)
    transformed = (values_f64 - mean) / scale
    projected = torch.einsum("bf,bkf->bk", transformed, components)
    return projected.to(dtype=torch.float32)


def _align_component_signs_batch(
    components: torch.Tensor,
    reference_components: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    aligned = components.clone()
    current_reference = reference_components.clone()
    limit = min(int(current_reference.shape[0]), int(aligned.shape[1]))

    for index in range(int(aligned.shape[0])):
        dot_products = torch.sum(
            aligned[index, :limit] * current_reference[:limit],
            dim=1,
        )
        flip_mask = dot_products < 0.0
        if torch.any(flip_mask):
            aligned[index, :limit][flip_mask] *= -1.0
        current_reference = aligned[index].clone()

    return aligned, current_reference


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
        device: Torch device for PCA math. ``"auto"`` prefers CUDA when
            available.
        batch_size: Number of walk-forward prefix windows to batch together per
            batched SVD call in the offline path.

    Example::

        transformer = WalkForwardPCATransformer(
            warmup=500,
            explained_variance_threshold=0.99,
            device="auto",
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
        device: str | torch.device = "auto",
        batch_size: int = _PCA_BATCH_SIZE,
    ) -> None:
        if warmup < 2:
            raise ValueError("warmup must be at least 2 for PCA.")
        if not 0.0 < explained_variance_threshold <= 1.0:
            raise ValueError(
                "explained_variance_threshold must be in the interval (0, 1]."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.warmup = int(warmup)
        self.explained_variance_threshold = float(explained_variance_threshold)
        self.standardize = bool(standardize)
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)

        self.n_features_in_: int | None = None
        self.n_components_: int | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.initial_explained_variance_ratio_: np.ndarray | None = None
        self._fit_state: _PCAFitState | None = None

    def _update_public_state(self, state: _PCAFitState) -> None:
        self.mean_ = _to_numpy_float32(state.mean)
        self.scale_ = _to_numpy_float32(state.scale)
        self.components_ = _to_numpy_float32(state.components)

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
        array = _as_2d_float_tensor(
            values,
            device=self.device,
            dtype=torch.float32,
        )
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
        self._fit_state = _PCAFitState(
            mean=initial_state.mean.clone(),
            scale=initial_state.scale.clone(),
            components=initial_state.components[:selected_components].clone(),
            explained_variance_ratio=initial_state.explained_variance_ratio.clone(),
        )
        self._update_public_state(self._fit_state)
        self.initial_explained_variance_ratio_ = _to_numpy_float32(
            initial_state.explained_variance_ratio
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
        if self._fit_state is None or self.components_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("WalkForwardPCATransformer must be fit before transform().")

        array = _as_2d_float_tensor(
            values,
            device=self.device,
            dtype=torch.float32,
        )
        if int(array.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"Expected {self.n_features_in_} input features, got {array.shape[1]}."
            )

        return _to_numpy_float32(_project_rows(array, self._fit_state))

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
        determine ``n_components_``. It then prepares chronological
        ``(history_stop, target_stop)`` windows once, processes those windows
        in chunks of ``batch_size``, refits PCA on rows ``[:history_stop]``,
        and projects only the corresponding next row. The output therefore
        starts at the first chronologically valid next-row projection.

        Args:
            values: 2D array-like input of shape ``(n_rows, n_features)`` in
                chronological order.
            progress_bar: If ``True``, show a ``tqdm`` progress bar over the
                chunked walk-forward loop. The bar still counts projected rows,
                even though updates happen chunk by chunk.

        Returns:
            A ``float32`` array of shape ``(n_rows - warmup, n_components_)``.

        Raises:
            ValueError: If the input is not a valid 2D matrix or contains fewer
                than ``warmup`` rows.
        """
        array = _as_2d_float_tensor(
            values,
            device=self.device,
            dtype=torch.float32,
        )
        self.fit(array)

        assert self.n_components_ is not None
        assert self._fit_state is not None

        windows = _make_walkforward_windows(int(array.shape[0]), self.warmup)
        if windows.size == 0:
            return np.empty((0, self.n_components_), dtype=np.float32)

        values_f64 = array.to(dtype=torch.float64)
        cumulative = torch.cumsum(values_f64, dim=0)
        cumulative_sq = (
            torch.cumsum(values_f64.square(), dim=0) if self.standardize else None
        )
        windows_t = torch.as_tensor(windows, device=self.device, dtype=torch.int64)

        projected_rows: list[torch.Tensor] = []
        reference_components = self._fit_state.components.clone()
        latest_state = self._fit_state

        total_rows = int(windows_t.shape[0])
        progress = _make_dataframe_progress_bar(total_rows) if progress_bar else None
        try:
            for start in range(0, total_rows, self.batch_size):
                stop = min(start + self.batch_size, total_rows)
                chunk = windows_t[start:stop]
                history_stops = chunk[:, 0]
                target_indices = chunk[:, 1] - 1

                mean, scale, components, explained_variance_ratio = _fit_pca_batch(
                    values_f64,
                    history_stops,
                    standardize=self.standardize,
                    n_components=self.n_components_,
                    cumulative=cumulative,
                    cumulative_sq=cumulative_sq,
                )
                aligned_components, reference_components = _align_component_signs_batch(
                    components,
                    reference_components,
                )

                projected_rows.append(
                    _project_rows_batch(
                        array[target_indices],
                        mean,
                        scale,
                        aligned_components,
                    )
                )
                latest_state = _PCAFitState(
                    mean=mean[-1].clone(),
                    scale=scale[-1].clone(),
                    components=aligned_components[-1].clone(),
                    explained_variance_ratio=explained_variance_ratio[-1].clone(),
                )
                if progress is not None:
                    progress.update(stop - start)
        finally:
            if progress is not None:
                progress.close()

        self._fit_state = latest_state
        self._update_public_state(latest_state)

        return _to_numpy_float32(torch.cat(projected_rows, dim=0))


def walkforward_pca_dataframe(
    df: Any,
    *,
    feature_columns: Sequence[str],
    warmup: int,
    explained_variance_threshold: float = 0.99,
    standardize: bool = True,
    device: str | torch.device = "auto",
    batch_size: int = _PCA_BATCH_SIZE,
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
        device: Torch device for PCA math. ``"auto"`` prefers CUDA when
            available.
        batch_size: Number of walk-forward prefix windows to batch together per
            batched SVD call.
        output_prefix: Prefix for appended PCA columns.
        drop_feature_columns: If ``True``, drop the original
            ``feature_columns`` after adding the PCA columns.
        trim_warmup: If ``True``, drop the leading warmup rows and reset the
            index so the returned dataframe contains only valid projections.
        progress_bar: If ``True``, show a ``tqdm`` progress bar over the
            chunked walk-forward PCA loop.

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
            device="auto",
            batch_size=256,
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
        device=device,
        batch_size=batch_size,
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

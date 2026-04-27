"""Walk-forward PCA helpers for time-series-safe dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch

from crosslearn._devices import resolve_device

_PCA_BATCH_SIZE = 256
_PCA_SOLVERS = ("svd", "covariance_eigh")
_PCA_COMPUTE_DTYPES = (torch.float32, torch.float64)


@dataclass
class _PCAFitState:
    mean: torch.Tensor
    scale: torch.Tensor
    components: torch.Tensor
    explained_variance_ratio: torch.Tensor


@dataclass
class _RunningStatisticsState:
    start: int
    stop: int
    sum_x: torch.Tensor
    sum_xx: torch.Tensor


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


def _validate_solver(solver: str) -> Literal["svd", "covariance_eigh"]:
    if solver not in _PCA_SOLVERS:
        raise ValueError(
            f"solver must be one of {_PCA_SOLVERS}, got {solver!r}."
        )
    return solver


def _validate_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _PCA_COMPUTE_DTYPES:
        raise ValueError(
            "compute_dtype must be torch.float32 or torch.float64."
        )
    return dtype


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


def _align_component_signs(
    components: torch.Tensor,
    reference_components: torch.Tensor | np.ndarray,
) -> torch.Tensor:
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

    aligned = components.clone()
    limit = min(int(reference_components.shape[0]), int(aligned.shape[0]))
    for index in range(limit):
        if float(torch.dot(aligned[index], reference_components[index]).item()) < 0.0:
            aligned[index] *= -1.0
    return aligned


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
            aligned_slice = aligned[index, :limit]
            aligned[index, :limit] = torch.where(
                flip_mask.unsqueeze(1),
                -aligned_slice,
                aligned_slice,
            )
        current_reference = aligned[index].clone()

    return aligned, current_reference


def _fit_pca_svd_from_history(
    history: torch.Tensor,
    *,
    standardize: bool,
    n_components: int | None = None,
) -> _PCAFitState:
    mean = history.mean(dim=0)
    transformed_history = history - mean

    if standardize:
        raw_scale = torch.std(history, dim=0, unbiased=False)
        scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
        transformed_history = transformed_history / scale
    else:
        scale = torch.ones(history.shape[1], dtype=history.dtype, device=history.device)

    _, singular_values, vt = torch.linalg.svd(transformed_history, full_matrices=False)
    variance_denom = max(int(history.shape[0]) - 1, 1)
    explained_variance = singular_values.square() / variance_denom
    total_variance = float(explained_variance.sum().item())
    if total_variance > 0.0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = torch.zeros_like(explained_variance)

    if n_components is None:
        n_components = int(vt.shape[0])

    return _PCAFitState(
        mean=mean,
        scale=scale,
        components=vt[:n_components].clone(),
        explained_variance_ratio=explained_variance_ratio,
    )


def _fit_pca_covariance_from_history(
    history: torch.Tensor,
    *,
    standardize: bool,
    n_components: int | None = None,
) -> _PCAFitState:
    mean = history.mean(dim=0)
    centered = history - mean

    if standardize:
        raw_scale = torch.std(history, dim=0, unbiased=False)
        scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
        transformed_history = centered / scale
    else:
        scale = torch.ones(history.shape[1], dtype=history.dtype, device=history.device)
        transformed_history = centered

    variance_denom = max(int(history.shape[0]) - 1, 1)
    covariance = (transformed_history.T @ transformed_history) / variance_denom
    covariance = (covariance + covariance.T) * 0.5

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    eigenvalues = torch.flip(eigenvalues, dims=(0,))
    eigenvectors = torch.flip(eigenvectors, dims=(1,))

    total_variance = float(eigenvalues.sum().item())
    if total_variance > 0.0:
        explained_variance_ratio = eigenvalues / total_variance
    else:
        explained_variance_ratio = torch.zeros_like(eigenvalues)

    if n_components is None:
        n_components = int(eigenvectors.shape[1])

    return _PCAFitState(
        mean=mean,
        scale=scale,
        components=eigenvectors.T[:n_components].clone(),
        explained_variance_ratio=explained_variance_ratio,
    )


def _fit_pca(
    history: torch.Tensor | np.ndarray,
    *,
    standardize: bool,
    n_components: int | None = None,
    reference_components: torch.Tensor | np.ndarray | None = None,
    solver: Literal["svd", "covariance_eigh"] = "svd",
    compute_dtype: torch.dtype = torch.float64,
) -> _PCAFitState:
    resolved_solver = _validate_solver(solver)
    resolved_compute_dtype = _validate_compute_dtype(compute_dtype)

    if not isinstance(history, torch.Tensor):
        history = _as_2d_float_tensor(
            history,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    history_compute = history.to(dtype=resolved_compute_dtype)
    if resolved_solver == "svd":
        fit_state = _fit_pca_svd_from_history(
            history_compute,
            standardize=standardize,
            n_components=n_components,
        )
    else:
        fit_state = _fit_pca_covariance_from_history(
            history_compute,
            standardize=standardize,
            n_components=n_components,
        )

    if reference_components is not None:
        fit_state = _PCAFitState(
            mean=fit_state.mean,
            scale=fit_state.scale,
            components=_align_component_signs(
                fit_state.components,
                reference_components,
            ),
            explained_variance_ratio=fit_state.explained_variance_ratio,
        )

    return fit_state


def _project_rows(values: torch.Tensor | np.ndarray, state: _PCAFitState) -> torch.Tensor:
    values_f32 = _as_2d_float_tensor(
        values,
        device=state.mean.device,
        dtype=torch.float32,
    )
    values_compute = values_f32.to(dtype=state.mean.dtype)
    transformed = (values_compute - state.mean) / state.scale
    projected = transformed @ state.components.T
    return projected.to(dtype=torch.float32)


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
    values_compute = values_f32.to(dtype=mean.dtype)
    transformed = (values_compute - mean) / scale
    projected = torch.einsum("bf,bkf->bk", transformed, components)
    return projected.to(dtype=torch.float32)


def _fit_pca_batch_svd_expanding(
    values_compute: torch.Tensor,
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
    base_history = values_compute[:max_history]
    history_lengths = history_stops.to(dtype=values_compute.dtype)

    if cumulative is None:
        cumulative = torch.cumsum(values_compute, dim=0)
    history_sum = cumulative[history_stops - 1]
    mean = history_sum / history_lengths.unsqueeze(1)

    if standardize:
        if cumulative_sq is None:
            cumulative_sq = torch.cumsum(values_compute.square(), dim=0)
        history_sq_sum = cumulative_sq[history_stops - 1]
        variance = history_sq_sum / history_lengths.unsqueeze(1) - mean.square()
        variance = torch.clamp(variance, min=0.0)
        raw_scale = torch.sqrt(variance)
        scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
    else:
        scale = torch.ones_like(mean)

    positions = torch.arange(max_history, device=values_compute.device)
    mask = positions.unsqueeze(0) < history_stops.unsqueeze(1)
    transformed_history = (
        (base_history.unsqueeze(0) - mean.unsqueeze(1)) / scale.unsqueeze(1)
    ) * mask.unsqueeze(-1).to(dtype=values_compute.dtype)

    _, singular_values, vt = torch.linalg.svd(transformed_history, full_matrices=False)
    explained_variance = singular_values.square()
    total_variance = explained_variance.sum(dim=1, keepdim=True)
    explained_variance_ratio = torch.where(
        total_variance > 0.0,
        explained_variance / total_variance,
        torch.zeros_like(explained_variance),
    )

    return mean, scale, vt[:, :n_components].clone(), explained_variance_ratio


def _fit_pca_batch_svd_rolling(
    values_compute: torch.Tensor,
    history_stops: torch.Tensor,
    *,
    warmup: int,
    standardize: bool,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history_starts = history_stops - warmup
    offsets = torch.arange(warmup, device=values_compute.device)
    history_indices = history_starts.unsqueeze(1) + offsets.unsqueeze(0)
    history_batch = values_compute[history_indices]

    mean = history_batch.mean(dim=1)
    transformed_history = history_batch - mean.unsqueeze(1)

    if standardize:
        raw_scale = torch.std(history_batch, dim=1, unbiased=False)
        scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
        transformed_history = transformed_history / scale.unsqueeze(1)
    else:
        scale = torch.ones_like(mean)

    _, singular_values, vt = torch.linalg.svd(transformed_history, full_matrices=False)
    explained_variance = singular_values.square()
    total_variance = explained_variance.sum(dim=1, keepdim=True)
    explained_variance_ratio = torch.where(
        total_variance > 0.0,
        explained_variance / total_variance,
        torch.zeros_like(explained_variance),
    )

    return mean, scale, vt[:, :n_components].clone(), explained_variance_ratio


def _initialize_running_statistics(
    values_compute: torch.Tensor,
    *,
    history_start: int,
    history_stop: int,
) -> _RunningStatisticsState:
    history = values_compute[history_start:history_stop]
    return _RunningStatisticsState(
        start=history_start,
        stop=history_stop,
        sum_x=history.sum(dim=0),
        sum_xx=history.T @ history,
    )


def _advance_running_statistics(
    state: _RunningStatisticsState,
    values_compute: torch.Tensor,
    *,
    history_start: int,
    history_stop: int,
) -> _RunningStatisticsState:
    while state.stop < history_stop:
        row = values_compute[state.stop]
        state.sum_x = state.sum_x + row
        state.sum_xx = state.sum_xx + torch.outer(row, row)
        state.stop += 1

    while state.start < history_start:
        row = values_compute[state.start]
        state.sum_x = state.sum_x - row
        state.sum_xx = state.sum_xx - torch.outer(row, row)
        state.start += 1

    return state


def _moments_to_mean_scale_covariance(
    state: _RunningStatisticsState,
    *,
    standardize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_length = state.stop - state.start
    if history_length < 1:
        raise ValueError("PCA history must contain at least one row.")

    history_length_float = torch.tensor(
        float(history_length),
        dtype=state.sum_x.dtype,
        device=state.sum_x.device,
    )
    mean = state.sum_x / history_length_float
    scatter = state.sum_xx - torch.outer(state.sum_x, state.sum_x) / history_length_float
    scatter = (scatter + scatter.T) * 0.5

    variance_denom = max(history_length - 1, 1)
    covariance = scatter / variance_denom

    if standardize:
        raw_variance = torch.diagonal(scatter) / history_length_float
        raw_variance = torch.clamp(raw_variance, min=0.0)
        raw_scale = torch.sqrt(raw_variance)
        scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
        covariance = covariance / torch.outer(scale, scale)
    else:
        scale = torch.ones_like(mean)

    covariance = (covariance + covariance.T) * 0.5
    return mean, scale, covariance


def _fit_pca_batch_covariance(
    values_compute: torch.Tensor,
    history_stops: torch.Tensor,
    *,
    warmup: int,
    standardize: bool,
    n_components: int,
    expanding_warmup: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if history_stops.ndim != 1:
        raise ValueError("history_stops must be a 1D tensor.")
    if history_stops.numel() == 0:
        raise ValueError("history_stops must contain at least one value.")

    batch_size = int(history_stops.shape[0])
    feature_count = int(values_compute.shape[1])
    mean_batch = torch.empty(
        (batch_size, feature_count),
        device=values_compute.device,
        dtype=values_compute.dtype,
    )
    scale_batch = torch.empty_like(mean_batch)
    covariance_batch = torch.empty(
        (batch_size, feature_count, feature_count),
        device=values_compute.device,
        dtype=values_compute.dtype,
    )

    first_stop = int(history_stops[0].item())
    first_start = 0 if expanding_warmup else first_stop - warmup
    state = _initialize_running_statistics(
        values_compute,
        history_start=first_start,
        history_stop=first_stop,
    )

    for batch_index, history_stop_value in enumerate(history_stops.tolist()):
        history_stop = int(history_stop_value)
        history_start = 0 if expanding_warmup else history_stop - warmup
        state = _advance_running_statistics(
            state,
            values_compute,
            history_start=history_start,
            history_stop=history_stop,
        )
        mean, scale, covariance = _moments_to_mean_scale_covariance(
            state,
            standardize=standardize,
        )
        mean_batch[batch_index] = mean
        scale_batch[batch_index] = scale
        covariance_batch[batch_index] = covariance

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_batch)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    eigenvalues = torch.flip(eigenvalues, dims=(1,))
    eigenvectors = torch.flip(eigenvectors, dims=(2,))
    total_variance = eigenvalues.sum(dim=1, keepdim=True)
    explained_variance_ratio = torch.where(
        total_variance > 0.0,
        eigenvalues / total_variance,
        torch.zeros_like(eigenvalues),
    )
    components = eigenvectors.transpose(1, 2)[:, :n_components].clone()
    return mean_batch, scale_batch, components, explained_variance_ratio


def _fit_pca_batch(
    values_compute: torch.Tensor,
    history_stops: torch.Tensor,
    *,
    warmup: int,
    standardize: bool,
    n_components: int,
    solver: Literal["svd", "covariance_eigh"],
    expanding_warmup: bool,
    cumulative: torch.Tensor | None = None,
    cumulative_sq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if solver == "svd":
        if expanding_warmup:
            return _fit_pca_batch_svd_expanding(
                values_compute,
                history_stops,
                standardize=standardize,
                n_components=n_components,
                cumulative=cumulative,
                cumulative_sq=cumulative_sq,
            )
        return _fit_pca_batch_svd_rolling(
            values_compute,
            history_stops,
            warmup=warmup,
            standardize=standardize,
            n_components=n_components,
        )

    return _fit_pca_batch_covariance(
        values_compute,
        history_stops,
        warmup=warmup,
        standardize=standardize,
        n_components=n_components,
        expanding_warmup=expanding_warmup,
    )


class WalkForwardPCATransformer:
    """Walk-forward PCA with configurable solver, history, and precision.

    ``fit(...)`` determines the fixed component count from the initial warmup
    window. ``walkforward_transform(...)`` then keeps that width fixed while
    refitting the centering, optional standardization, and PCA loadings on
    either expanding or rolling history before projecting the next row.

    Args:
        warmup: Number of initial rows used to choose the fixed PCA width and to
            fit the first PCA state. Must be at least ``2``.
        explained_variance_threshold: Cumulative explained-variance threshold
            used on the initial warmup fit to choose ``n_components_``.
        standardize: If ``True``, center and divide by the walk-forward
            standard deviation before PCA. If ``False``, PCA is still centered
            but not variance-scaled.
        solver: PCA backend. ``"svd"`` uses direct singular-value
            decomposition. ``"covariance_eigh"`` decomposes the covariance or
            correlation matrix derived from the selected history window.
        expanding_warmup: If ``True``, fit PCA on all available past rows after
            warmup. If ``False``, fit PCA on exactly the last ``warmup`` rows
            before each next-row projection.
        compute_dtype: Internal torch dtype used for PCA math. ``float64`` is
            the default stability path, while ``float32`` is an opt-in faster
            path that may be preferable on CUDA.
        device: Torch device for PCA math. ``"auto"`` prefers CUDA when
            available.
        batch_size: Number of chronological PCA windows to process per offline
            chunk.

    Example::

        transformer = WalkForwardPCATransformer(
            warmup=500,
            explained_variance_threshold=0.99,
            solver="svd",
            expanding_warmup=True,
            compute_dtype=torch.float64,
            device="auto",
            batch_size=256,
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
        solver: Literal["svd", "covariance_eigh"] = "svd",
        expanding_warmup: bool = True,
        compute_dtype: torch.dtype = torch.float64,
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
        self.solver = _validate_solver(solver)
        self.expanding_warmup = bool(expanding_warmup)
        self.compute_dtype = _validate_compute_dtype(compute_dtype)
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

        Only the first ``warmup`` rows are used here. This method does not run
        the full walk-forward projection loop.

        Args:
            values: 2D array-like input of shape ``(n_rows, n_features)``.

        Returns:
            ``self`` with the initial PCA state populated.

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

        initial_state = _fit_pca(
            array[: self.warmup],
            standardize=self.standardize,
            solver=self.solver,
            compute_dtype=self.compute_dtype,
        )
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
        mean, scale, and loadings to every row in ``values``.

        Args:
            values: 2D array-like input of shape ``(n_rows, n_features)``.

        Returns:
            A ``float32`` array of shape ``(n_rows, n_components_)``.

        Raises:
            ValueError: If the transformer has not been fit or if the feature
                count does not match the fitted state.
        """
        if (
            self._fit_state is None
            or self.components_ is None
            or self.mean_ is None
            or self.scale_ is None
        ):
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
            A ``float32`` array of shape ``(n_rows - warmup, n_components_)``.
        """
        return self.walkforward_transform(values)

    def walkforward_transform(
        self,
        values: Any,
        *,
        progress_bar: bool = False,
    ) -> np.ndarray:
        """Project each row using PCA refit on past rows only.

        The method first calls ``fit(...)`` on the initial warmup window to
        determine ``n_components_``. It then prepares chronological
        ``(history_stop, target_stop)`` windows once, processes those windows
        in chunks of ``batch_size``, refits PCA on the selected history window,
        and projects only the corresponding next row.

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

        values_compute = array.to(dtype=self.compute_dtype)
        windows_t = torch.as_tensor(windows, device=self.device, dtype=torch.int64)

        cumulative = None
        cumulative_sq = None
        if self.solver == "svd" and self.expanding_warmup:
            cumulative = torch.cumsum(values_compute, dim=0)
            cumulative_sq = (
                torch.cumsum(values_compute.square(), dim=0)
                if self.standardize
                else None
            )

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
                    values_compute,
                    history_stops,
                    warmup=self.warmup,
                    standardize=self.standardize,
                    n_components=self.n_components_,
                    solver=self.solver,
                    expanding_warmup=self.expanding_warmup,
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
    solver: Literal["svd", "covariance_eigh"] = "svd",
    expanding_warmup: bool = True,
    compute_dtype: torch.dtype = torch.float64,
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

    This function composes cleanly with ``embed_dataframe(...)``: first build
    Chronos embeddings, then run walk-forward PCA over the resulting
    ``chronos_*`` columns.

    Args:
        df: Source pandas dataframe in chronological order.
        feature_columns: Numeric columns to reduce with walk-forward PCA.
        warmup: Number of initial rows used to determine the fixed PCA width.
        explained_variance_threshold: Initial cumulative explained-variance
            threshold used to choose ``n_components_``.
        standardize: If ``True``, recompute mean/std walk-forward before each
            PCA fit. If ``False``, only walk-forward centering is applied.
        solver: PCA backend. ``"svd"`` is the direct decomposition path.
            ``"covariance_eigh"`` solves PCA from a square covariance or
            correlation matrix derived from each history window.
        expanding_warmup: If ``True``, fit PCA on all available past rows after
            warmup. If ``False``, fit PCA on exactly the last ``warmup`` rows
            before each next-row projection.
        compute_dtype: Internal torch dtype used for PCA math.
        device: Torch device for PCA math. ``"auto"`` prefers CUDA when
            available.
        batch_size: Number of chronological PCA windows to process per offline
            chunk.
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
            solver="svd",
            expanding_warmup=True,
            compute_dtype=torch.float64,
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
        solver=solver,
        expanding_warmup=expanding_warmup,
        compute_dtype=compute_dtype,
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

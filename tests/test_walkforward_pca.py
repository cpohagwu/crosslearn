from __future__ import annotations

import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

import crosslearn.extractors.pca as pca_module
from crosslearn.envs import WalkForwardChronosPCAWrapper
from crosslearn.extractors import (
    WalkForwardPCATransformer,
    embed_dataframe,
    walkforward_pca_dataframe,
)
from crosslearn.extractors.chronos import ChronosEmbedder


def _install_fake_tqdm(monkeypatch):
    instances = []

    class _FakeTqdm:
        def __init__(self, *, total, unit, desc, dynamic_ncols) -> None:
            self.total = total
            self.unit = unit
            self.desc = desc
            self.dynamic_ncols = dynamic_ncols
            self.updates: list[int] = []
            self.closed = False

        def update(self, value: int) -> None:
            self.updates.append(value)

        def close(self) -> None:
            self.closed = True

    module = types.ModuleType("tqdm")
    auto_module = types.ModuleType("tqdm.auto")

    def _tqdm(*, total, unit, desc, dynamic_ncols):
        bar = _FakeTqdm(
            total=total,
            unit=unit,
            desc=desc,
            dynamic_ncols=dynamic_ncols,
        )
        instances.append(bar)
        return bar

    module.tqdm = _tqdm
    module.auto = auto_module
    auto_module.tqdm = _tqdm
    monkeypatch.setitem(sys.modules, "tqdm", module)
    monkeypatch.setitem(sys.modules, "tqdm.auto", auto_module)
    return instances


def _make_pca_dataframe():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0],
            "Close": [2.0, 4.0, 8.0, 14.0, 22.0, 32.0, 44.0, 58.0],
            "Volume": [10.0, 11.0, 13.0, 16.0, 20.0, 25.0, 31.0, 38.0],
        }
    )


def _manual_initial_projection(
    values: np.ndarray,
    *,
    warmup: int,
    explained_variance_threshold: float,
    standardize: bool,
    solver: str = "svd",
) -> tuple[np.ndarray, int]:
    history = values[:warmup].astype(np.float64)
    target = values[warmup].astype(np.float64)
    return _manual_projection_from_history(
        history,
        target,
        explained_variance_threshold=explained_variance_threshold,
        standardize=standardize,
        solver=solver,
    )


def _manual_projection_from_history(
    history: np.ndarray,
    target: np.ndarray,
    *,
    explained_variance_threshold: float,
    standardize: bool,
    solver: str = "svd",
) -> tuple[np.ndarray, int]:
    history = history.astype(np.float64)
    target = target.astype(np.float64)

    mean = history.mean(axis=0)
    history_transformed = history - mean
    target_transformed = target - mean

    if standardize:
        scale = history.std(axis=0, ddof=0)
        safe_scale = np.where(scale > 0.0, scale, 1.0)
        history_transformed = history_transformed / safe_scale
        target_transformed = target_transformed / safe_scale

    if solver == "svd":
        _, singular_values, vt = np.linalg.svd(history_transformed, full_matrices=False)
        explained_variance = (singular_values**2) / max(history.shape[0] - 1, 1)
        components = vt
    elif solver == "covariance_eigh":
        covariance = (history_transformed.T @ history_transformed) / max(
            history.shape[0] - 1,
            1,
        )
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        explained_variance = np.clip(eigenvalues[::-1], a_min=0.0, a_max=None)
        components = eigenvectors[:, ::-1].T
    else:
        raise ValueError(f"Unsupported solver for test helper: {solver!r}")

    total_variance = float(explained_variance.sum())
    explained_variance_ratio = explained_variance / total_variance
    n_components = int(
        np.searchsorted(
            np.cumsum(explained_variance_ratio, dtype=np.float64),
            explained_variance_threshold,
            side="left",
        )
        + 1
    )
    components = components[:n_components]
    return (target_transformed @ components.T).astype(np.float32), n_components


def _align_projection_signs(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    aligned = candidate.copy()
    limit = min(reference.shape[1], aligned.shape[1])
    for component_index in range(limit):
        if np.dot(reference[:, component_index], aligned[:, component_index]) < 0.0:
            aligned[:, component_index] *= -1.0
    return aligned


def test_make_walkforward_windows_builds_history_target_pairs() -> None:
    windows = pca_module._make_walkforward_windows(total_rows=8, warmup=3)

    np.testing.assert_array_equal(
        windows[:4],
        np.array(
            [
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
            ],
            dtype=np.int64,
        ),
    )


class _SequentialWindowEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        *,
        df,
        feature_columns: list[str],
        window_size: int,
        frame_bound: tuple[int, int],
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        self.window_size = int(window_size)
        self.frame_bound = (int(frame_bound[0]), int(frame_bound[1]))
        self._max_steps = self.frame_bound[1] - self.frame_bound[0]
        self._position = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.feature_columns)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

    def _window(self) -> np.ndarray:
        end = self.frame_bound[0] + self._position
        start = end - self.window_size
        return self.df.iloc[start:end][self.feature_columns].to_numpy(dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._position = 0
        return self._window(), {}

    def step(self, action: int):
        self._position = min(self._position + 1, self._max_steps)
        terminated = self._position >= self._max_steps
        return self._window(), float(action), terminated, False, {"position": self._position}


def test_walkforward_pca_transformer_matches_manual_first_projection() -> None:
    values = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 11.0, 5.0],
            [4.0, 13.0, 9.0],
            [7.0, 16.0, 15.0],
            [11.0, 20.0, 23.0],
        ],
        dtype=np.float32,
    )
    transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        device="cpu",
        batch_size=1,
    )

    projected = transformer.walkforward_transform(values)
    expected_first, expected_n_components = _manual_initial_projection(
        values,
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
    )

    assert transformer.n_components_ == expected_n_components
    np.testing.assert_allclose(projected[0], expected_first, atol=1e-5)


def test_walkforward_pca_transformer_center_only_matches_manual_first_projection() -> None:
    values = np.array(
        [
            [1.0, 1.0, 10.0],
            [2.0, 3.0, 11.0],
            [4.0, 8.0, 13.0],
            [8.0, 21.0, 16.0],
            [16.0, 55.0, 20.0],
        ],
        dtype=np.float32,
    )
    transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.9,
        standardize=False,
        device="cpu",
        batch_size=1,
    )

    projected = transformer.walkforward_transform(values)
    expected_first, expected_n_components = _manual_initial_projection(
        values,
        warmup=3,
        explained_variance_threshold=0.9,
        standardize=False,
    )

    assert transformer.n_components_ == expected_n_components
    np.testing.assert_allclose(projected[0], expected_first, atol=1e-5)


def test_walkforward_pca_transformer_batching_matches_rowwise_cpu() -> None:
    values = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 11.0, 5.0],
            [4.0, 13.0, 9.0],
            [7.0, 16.0, 15.0],
            [11.0, 20.0, 23.0],
            [16.0, 25.0, 33.0],
            [22.0, 31.0, 45.0],
        ],
        dtype=np.float32,
    )
    rowwise = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        device="cpu",
        batch_size=1,
    )
    batched = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        device="cpu",
        batch_size=3,
    )

    rowwise_projected = rowwise.walkforward_transform(values)
    batched_projected = batched.walkforward_transform(values)

    np.testing.assert_allclose(batched_projected, rowwise_projected, atol=1e-5)


def test_walkforward_pca_transformer_covariance_solver_matches_svd() -> None:
    values = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 11.0, 5.0],
            [4.0, 13.0, 9.0],
            [7.0, 16.0, 15.0],
            [11.0, 20.0, 23.0],
            [16.0, 25.0, 33.0],
            [22.0, 31.0, 45.0],
        ],
        dtype=np.float32,
    )
    svd_transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver="svd",
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
    )
    covariance_transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver="covariance_eigh",
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
    )

    svd_projected = svd_transformer.walkforward_transform(values)
    covariance_projected = covariance_transformer.walkforward_transform(values)
    aligned_covariance = _align_projection_signs(svd_projected, covariance_projected)

    assert covariance_transformer.n_components_ == svd_transformer.n_components_
    np.testing.assert_allclose(
        aligned_covariance,
        svd_projected,
        atol=1e-4,
    )


def test_walkforward_pca_transformer_rolling_matches_manual_projection() -> None:
    values = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 11.0, 5.0],
            [4.0, 13.0, 9.0],
            [7.0, 16.0, 15.0],
            [11.0, 20.0, 23.0],
            [16.0, 25.0, 33.0],
        ],
        dtype=np.float32,
    )
    transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver="svd",
        expanding_warmup=False,
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=1,
    )

    projected = transformer.walkforward_transform(values)
    expected_first, expected_n_components = _manual_projection_from_history(
        values[:3],
        values[3],
        explained_variance_threshold=0.95,
        standardize=True,
        solver="svd",
    )
    expected_second, _ = _manual_projection_from_history(
        values[1:4],
        values[4],
        explained_variance_threshold=0.95,
        standardize=True,
        solver="svd",
    )

    assert transformer.n_components_ == expected_n_components
    np.testing.assert_allclose(projected[0], expected_first, atol=1e-5)
    np.testing.assert_allclose(projected[1], expected_second, atol=1e-5)


def test_walkforward_pca_transformer_rolling_covariance_solver_matches_svd() -> None:
    values = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 11.0, 5.0],
            [4.0, 13.0, 9.0],
            [7.0, 16.0, 15.0],
            [11.0, 20.0, 23.0],
            [16.0, 25.0, 33.0],
            [22.0, 31.0, 45.0],
        ],
        dtype=np.float32,
    )
    svd_transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver="svd",
        expanding_warmup=False,
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
    )
    covariance_transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver="covariance_eigh",
        expanding_warmup=False,
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
    )

    svd_projected = svd_transformer.walkforward_transform(values)
    covariance_projected = covariance_transformer.walkforward_transform(values)
    aligned_covariance = _align_projection_signs(svd_projected, covariance_projected)

    assert covariance_transformer.n_components_ == svd_transformer.n_components_
    np.testing.assert_allclose(
        aligned_covariance,
        svd_projected,
        atol=1e-4,
    )


def test_walkforward_pca_transformer_float32_compute_matches_float64() -> None:
    values = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 11.0, 5.0],
            [4.0, 13.0, 9.0],
            [7.0, 16.0, 15.0],
            [11.0, 20.0, 23.0],
            [16.0, 25.0, 33.0],
            [22.0, 31.0, 45.0],
        ],
        dtype=np.float32,
    )
    float64_transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver="covariance_eigh",
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
    )
    float32_transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver="covariance_eigh",
        compute_dtype=torch.float32,
        device="cpu",
        batch_size=2,
    )

    projected64 = float64_transformer.walkforward_transform(values)
    projected32 = float32_transformer.walkforward_transform(values)
    aligned32 = _align_projection_signs(projected64, projected32)

    np.testing.assert_allclose(aligned32, projected64, atol=1e-4)


def test_walkforward_pca_dataframe_appends_and_trims_columns() -> None:
    df = _make_pca_dataframe()

    full = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        output_prefix="pca_",
        drop_feature_columns=False,
        trim_warmup=False,
    )
    full_with_nan_warmup = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        output_prefix="pca_",
        drop_feature_columns=False,
        return_transformed_warmup=False,
        trim_warmup=False,
    )
    trimmed = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        output_prefix="pca_",
        drop_feature_columns=True,
        trim_warmup=True,
    )
    trimmed_without_warmup_scores = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        output_prefix="pca_",
        drop_feature_columns=True,
        return_transformed_warmup=False,
        trim_warmup=True,
    )

    pca_columns = [column for column in full.columns if column.startswith("pca_")]
    assert pca_columns
    assert not full.loc[:2, pca_columns].isna().any().any()
    assert full_with_nan_warmup.loc[:2, pca_columns].isna().all().all()
    np.testing.assert_allclose(
        full.loc[3:, pca_columns].to_numpy(dtype=np.float32),
        full_with_nan_warmup.loc[3:, pca_columns].to_numpy(dtype=np.float32),
        atol=1e-5,
    )
    assert len(trimmed) == len(df) - 3
    assert not trimmed[pca_columns].isna().any().any()
    assert set(trimmed.columns) == set(pca_columns)
    np.testing.assert_allclose(
        trimmed.to_numpy(dtype=np.float32),
        trimmed_without_warmup_scores.to_numpy(dtype=np.float32),
        atol=1e-5,
    )


def test_walkforward_pca_dataframe_returns_initial_warmup_fit_transform() -> None:
    df = _make_pca_dataframe()
    values = df.loc[:, ["Open", "Close", "Volume"]].to_numpy(dtype=np.float32)

    transformed = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        batch_size=2,
        trim_warmup=False,
    )
    pca_columns = [column for column in transformed.columns if column.startswith("pca_")]

    transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.99,
        standardize=True,
        device="cpu",
        batch_size=2,
    )
    transformer.fit(values)
    expected_warmup = transformer.transform(values[:3])

    np.testing.assert_allclose(
        transformed.loc[:2, pca_columns].to_numpy(dtype=np.float32),
        expected_warmup,
        atol=1e-5,
    )


def test_walkforward_pca_dataframe_exact_warmup_length_handles_trim_modes() -> None:
    df = _make_pca_dataframe().iloc[:3].reset_index(drop=True)

    full = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        trim_warmup=False,
    )
    trimmed = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        trim_warmup=True,
    )

    pca_columns = [column for column in full.columns if column.startswith("pca_")]
    assert len(full) == 3
    assert pca_columns
    assert not full[pca_columns].isna().any().any()
    assert len(trimmed) == 0
    assert list(trimmed.columns) == list(full.columns)


def test_walkforward_pca_dataframe_batching_matches_rowwise_cpu() -> None:
    df = _make_pca_dataframe()

    rowwise = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        batch_size=1,
        trim_warmup=True,
    )
    batched = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        batch_size=3,
        trim_warmup=True,
    )

    np.testing.assert_allclose(
        batched.filter(like="pca_").to_numpy(dtype=np.float32),
        rowwise.filter(like="pca_").to_numpy(dtype=np.float32),
        atol=1e-5,
    )


def test_walkforward_pca_dataframe_covariance_solver_matches_svd() -> None:
    df = _make_pca_dataframe()

    svd_result = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        solver="svd",
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
        trim_warmup=True,
    )
    covariance_result = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        solver="covariance_eigh",
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
        trim_warmup=True,
    )

    np.testing.assert_allclose(
        _align_projection_signs(
            svd_result.filter(like="pca_").to_numpy(dtype=np.float32),
            covariance_result.filter(like="pca_").to_numpy(dtype=np.float32),
        ),
        svd_result.filter(like="pca_").to_numpy(dtype=np.float32),
        atol=1e-4,
    )


def test_walkforward_pca_dataframe_rolling_matches_rowwise_cpu() -> None:
    df = _make_pca_dataframe()

    rowwise = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        solver="svd",
        expanding_warmup=False,
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=1,
        trim_warmup=True,
    )
    batched = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        solver="svd",
        expanding_warmup=False,
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=3,
        trim_warmup=True,
    )

    np.testing.assert_allclose(
        batched.filter(like="pca_").to_numpy(dtype=np.float32),
        rowwise.filter(like="pca_").to_numpy(dtype=np.float32),
        atol=1e-5,
    )


def test_walkforward_pca_dataframe_progress_bar_matches_default(
    monkeypatch,
) -> None:
    df = _make_pca_dataframe()
    _install_fake_tqdm(monkeypatch)

    base = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        batch_size=2,
    )
    with_progress = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        batch_size=2,
        progress_bar=True,
    )

    np.testing.assert_allclose(
        with_progress.filter(like="pca_").to_numpy(dtype=np.float32),
        base.filter(like="pca_").to_numpy(dtype=np.float32),
        equal_nan=True,
    )


def test_walkforward_pca_dataframe_progress_bar_tracks_rows(monkeypatch) -> None:
    df = _make_pca_dataframe()
    bars = _install_fake_tqdm(monkeypatch)

    transformed = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        device="cpu",
        batch_size=2,
        progress_bar=True,
    )

    assert transformed.filter(like="pca_").shape[1] >= 1
    assert len(bars) == 1
    assert bars[0].total == len(df) - 3
    assert bars[0].unit == "row"
    assert bars[0].desc == "Walk-forward PCA"
    assert bars[0].dynamic_ncols is True
    assert bars[0].updates == [2, 2, 1]
    assert bars[0].closed is True


def test_walkforward_pca_dataframe_progress_bar_requires_tqdm(
    monkeypatch,
) -> None:
    df = _make_pca_dataframe()

    def _raise_import_error():
        raise ImportError("tqdm is not installed")

    original_import = __import__

    def _patched_import(name, *args, **kwargs):
        if name in {"tqdm", "tqdm.auto"}:
            _raise_import_error()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _patched_import)
    monkeypatch.delitem(sys.modules, "tqdm", raising=False)
    monkeypatch.delitem(sys.modules, "tqdm.auto", raising=False)

    with pytest.raises(ImportError, match="tqdm"):
        walkforward_pca_dataframe(
            df,
            feature_columns=["Open", "Close", "Volume"],
            warmup=3,
            standardize=True,
            device="cpu",
            progress_bar=True,
        )


def test_walkforward_pca_sign_alignment_stabilizes_component_orientation(
    monkeypatch,
) -> None:
    original_svd = pca_module.torch.linalg.svd
    call_count = {"value": 0}

    def _patched_svd(*args, **kwargs):
        u, singular_values, vt = original_svd(*args, **kwargs)
        if call_count["value"] % 2 == 1:
            vt = vt.clone()
            vt[0] *= -1.0
        call_count["value"] += 1
        return u, singular_values, vt

    monkeypatch.setattr(pca_module.torch.linalg, "svd", _patched_svd)

    values = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
        ],
        dtype=np.float32,
    )
    transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.9,
        standardize=False,
        device="cpu",
    )

    projected = transformer.walkforward_transform(values)
    orientation = np.sign(projected[0, 0])

    assert orientation != 0.0
    np.testing.assert_array_equal(
        np.sign(projected[:, 0]),
        np.full(projected.shape[0], orientation, dtype=np.float32),
    )


def test_walkforward_pca_transformer_auto_resolves_to_cpu_when_cuda_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.9,
        standardize=True,
        device="auto",
    )

    assert transformer.device.type == "cpu"


@pytest.mark.parametrize(
    ("solver", "expanding_warmup"),
    [
        ("svd", True),
        ("svd", False),
        ("covariance_eigh", True),
        ("covariance_eigh", False),
    ],
)
def test_walkforward_chronos_pca_wrapper_matches_explicit_offline_pipeline(
    fake_chronos,
    solver: str,
    expanding_warmup: bool,
) -> None:
    df = _make_pca_dataframe()
    lookback = 3
    warmup = 3
    feature_columns = ["Open", "Close", "Volume"]
    agent_frame_bound = (lookback + warmup, len(df))
    history_frame_bound = (agent_frame_bound[0] - warmup, agent_frame_bound[1])

    embedded = embed_dataframe(
        df,
        lookback=lookback,
        frame_bound=history_frame_bound,
        feature_columns=feature_columns,
        selected_columns=["Close", "Volume"],
    )
    expected_df = walkforward_pca_dataframe(
        embedded,
        feature_columns=[column for column in embedded.columns if column.startswith("chronos_")],
        warmup=warmup,
        explained_variance_threshold=0.99,
        standardize=True,
        solver=solver,
        expanding_warmup=expanding_warmup,
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
        output_prefix="pca_",
        drop_feature_columns=True,
        trim_warmup=True,
    )
    expected = expected_df.filter(like="pca_").to_numpy(dtype=np.float32)

    env = _SequentialWindowEnv(
        df=df,
        feature_columns=feature_columns,
        window_size=lookback,
        frame_bound=agent_frame_bound,
    )
    wrapped = WalkForwardChronosPCAWrapper(
        env,
        lookback=lookback,
        warmup=warmup,
        feature_columns=feature_columns,
        selected_columns=["Close", "Volume"],
        solver=solver,
        expanding_warmup=expanding_warmup,
        compute_dtype=torch.float64,
        device_map="cpu",
    )

    observations = []
    obs, _ = wrapped.reset()
    np.testing.assert_allclose(obs, expected[0], atol=1e-5)
    observations.append(obs)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = wrapped.step(0)
        observations.append(obs)

    np.testing.assert_allclose(
        np.asarray(observations, dtype=np.float32),
        expected,
        atol=1e-5,
    )
    assert wrapped.observation_space.shape == (expected.shape[1],)


@pytest.mark.parametrize(
    ("df_length", "frame_bound", "match"),
    [
        (6, (5, 6), r"frame_bound\[0\] must be at least lookback \+ warmup"),
        (6, (6, 6), r"frame_bound\[1\] must be greater than frame_bound\[0\]"),
        (6, (6, 5), r"frame_bound\[1\] must be greater than frame_bound\[0\]"),
    ],
)
def test_walkforward_chronos_pca_wrapper_validates_boundary_requirements(
    fake_chronos,
    df_length: int,
    frame_bound: tuple[int, int],
    match: str,
) -> None:
    df = _make_pca_dataframe().iloc[:df_length].reset_index(drop=True)
    env = _SequentialWindowEnv(
        df=df,
        feature_columns=["Open", "Close", "Volume"],
        window_size=3,
        frame_bound=frame_bound,
    )

    with pytest.raises(ValueError, match=match):
        WalkForwardChronosPCAWrapper(
            env,
            lookback=3,
            warmup=3,
            feature_columns=["Open", "Close", "Volume"],
            selected_columns=["Close", "Volume"],
            device_map="cpu",
        )


def test_walkforward_chronos_pca_wrapper_accepts_exact_minimum_dataset(
    fake_chronos,
) -> None:
    df = _make_pca_dataframe().iloc[:7].reset_index(drop=True)
    lookback = 3
    warmup = 3
    env = _SequentialWindowEnv(
        df=df,
        feature_columns=["Open", "Close", "Volume"],
        window_size=lookback,
        frame_bound=(lookback + warmup, len(df)),
    )
    wrapped = WalkForwardChronosPCAWrapper(
        env,
        lookback=lookback,
        warmup=warmup,
        feature_columns=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
        device_map="cpu",
    )

    obs, _ = wrapped.reset()

    assert obs.shape == wrapped.observation_space.shape
    assert fake_chronos.last_pipeline is not None
    assert [tuple(call.shape) for call in fake_chronos.last_pipeline.calls] == [
        (warmup, lookback, 2),
        (1, lookback, 2),
    ]


def test_walkforward_chronos_pca_wrapper_embeds_only_one_new_window_per_step(
    fake_chronos,
) -> None:
    df = _make_pca_dataframe()
    lookback = 3
    warmup = 3

    env = _SequentialWindowEnv(
        df=df,
        feature_columns=["Open", "Close", "Volume"],
        window_size=lookback,
        frame_bound=(lookback + warmup, len(df)),
    )
    wrapped = WalkForwardChronosPCAWrapper(
        env,
        lookback=lookback,
        warmup=warmup,
        feature_columns=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
        device_map="cpu",
    )

    assert fake_chronos.last_pipeline is not None
    assert [tuple(call.shape) for call in fake_chronos.last_pipeline.calls] == [
        (warmup, lookback, 2),
    ]

    wrapped.reset()
    assert [tuple(call.shape) for call in fake_chronos.last_pipeline.calls] == [
        (warmup, lookback, 2),
        (1, lookback, 2),
    ]

    wrapped.step(0)
    wrapped.step(0)
    assert [tuple(call.shape) for call in fake_chronos.last_pipeline.calls] == [
        (warmup, lookback, 2),
        (1, lookback, 2),
        (1, lookback, 2),
        (1, lookback, 2),
    ]


def test_walkforward_chronos_pca_wrapper_requests_tensor_embeddings_on_cpu(
    fake_chronos,
    monkeypatch,
) -> None:
    recorded: list[tuple[bool | None, torch.device | str | None]] = []
    original_embed_windows = ChronosEmbedder.embed_windows

    def _patched_embed_windows(self, *args, **kwargs):
        recorded.append((kwargs.get("as_tensor"), kwargs.get("output_device")))
        return original_embed_windows(self, *args, **kwargs)

    monkeypatch.setattr(ChronosEmbedder, "embed_windows", _patched_embed_windows)

    df = _make_pca_dataframe()
    lookback = 3
    warmup = 3
    env = _SequentialWindowEnv(
        df=df,
        feature_columns=["Open", "Close", "Volume"],
        window_size=lookback,
        frame_bound=(lookback + warmup, len(df)),
    )
    wrapped = WalkForwardChronosPCAWrapper(
        env,
        lookback=lookback,
        warmup=warmup,
        feature_columns=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
        device_map="cpu",
        pca_device="cpu",
    )

    wrapped.reset()

    assert recorded[0][0] is True
    assert torch.device(recorded[0][1]).type == "cpu"
    assert recorded[1][0] is True
    assert torch.device(recorded[1][1]).type == "cpu"
    assert isinstance(wrapped._warmup_embeddings, torch.Tensor)
    assert isinstance(wrapped._current_embedding, torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_walkforward_chronos_pca_wrapper_supports_split_devices(
    fake_chronos,
    monkeypatch,
) -> None:
    recorded: list[tuple[bool | None, torch.device | str | None]] = []
    original_embed_windows = ChronosEmbedder.embed_windows

    def _patched_embed_windows(self, *args, **kwargs):
        recorded.append((kwargs.get("as_tensor"), kwargs.get("output_device")))
        return original_embed_windows(self, *args, **kwargs)

    monkeypatch.setattr(ChronosEmbedder, "embed_windows", _patched_embed_windows)

    df = _make_pca_dataframe()
    env = _SequentialWindowEnv(
        df=df,
        feature_columns=["Open", "Close", "Volume"],
        window_size=3,
        frame_bound=(6, len(df)),
    )
    wrapped = WalkForwardChronosPCAWrapper(
        env,
        lookback=3,
        warmup=3,
        feature_columns=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
        device_map="cuda",
        pca_device="cpu",
    )

    wrapped.reset()

    assert recorded[0][0] is True
    assert torch.device(recorded[0][1]).type == "cpu"
    assert recorded[1][0] is True
    assert torch.device(recorded[1][1]).type == "cpu"


@pytest.mark.parametrize("solver", ["svd", "covariance_eigh"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_walkforward_pca_transformer_cuda_matches_cpu(solver: str) -> None:
    values = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 11.0, 5.0],
            [4.0, 13.0, 9.0],
            [7.0, 16.0, 15.0],
            [11.0, 20.0, 23.0],
            [16.0, 25.0, 33.0],
        ],
        dtype=np.float32,
    )

    cpu_transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver=solver,
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
    )
    cuda_transformer = WalkForwardPCATransformer(
        warmup=3,
        explained_variance_threshold=0.95,
        standardize=True,
        solver=solver,
        compute_dtype=torch.float64,
        device="cuda",
        batch_size=2,
    )

    cpu_projected = cpu_transformer.walkforward_transform(values)
    cuda_projected = cuda_transformer.walkforward_transform(values)
    aligned_cuda = _align_projection_signs(cpu_projected, cuda_projected)

    np.testing.assert_allclose(aligned_cuda, cpu_projected, atol=1e-5)


@pytest.mark.parametrize("solver", ["svd", "covariance_eigh"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_walkforward_pca_dataframe_cuda_matches_cpu(solver: str) -> None:
    df = _make_pca_dataframe()

    cpu_result = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        solver=solver,
        compute_dtype=torch.float64,
        device="cpu",
        batch_size=2,
        trim_warmup=True,
    )
    cuda_result = walkforward_pca_dataframe(
        df,
        feature_columns=["Open", "Close", "Volume"],
        warmup=3,
        standardize=True,
        solver=solver,
        compute_dtype=torch.float64,
        device="cuda",
        batch_size=2,
        trim_warmup=True,
    )

    np.testing.assert_allclose(
        _align_projection_signs(
            cpu_result.filter(like="pca_").to_numpy(dtype=np.float32),
            cuda_result.filter(like="pca_").to_numpy(dtype=np.float32),
        ),
        cpu_result.filter(like="pca_").to_numpy(dtype=np.float32),
        atol=1e-5,
    )


@pytest.mark.parametrize("solver", ["svd", "covariance_eigh"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_walkforward_chronos_pca_wrapper_cuda_returns_numpy_observations(
    fake_chronos,
    solver: str,
) -> None:
    df = _make_pca_dataframe()
    lookback = 3
    warmup = 3

    env = _SequentialWindowEnv(
        df=df,
        feature_columns=["Open", "Close", "Volume"],
        window_size=lookback,
        frame_bound=(lookback + warmup, len(df)),
    )
    wrapped = WalkForwardChronosPCAWrapper(
        env,
        lookback=lookback,
        warmup=warmup,
        feature_columns=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
        solver=solver,
        compute_dtype=torch.float64,
        device_map="cuda",
    )

    obs, _ = wrapped.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32

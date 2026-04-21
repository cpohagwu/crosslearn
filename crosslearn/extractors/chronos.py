"""Chronos-backed extractors and embedders for rolling time-series windows.

The utilities in this module are original package components for reusable
Chronos-backed RL features. They are not an implementation of ChronosRL
(Lima, Oliveira, and Zanchettin, 2025), though that paper is relevant adjacent
inspiration for Chronos-based reinforcement learning on market data.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, TypeAlias

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from crosslearn._devices import resolve_device_map
from crosslearn.extractors.base import BaseFeaturesExtractor

PoolingMode: TypeAlias = Literal["mean", "last"]
WindowInput: TypeAlias = np.ndarray | torch.Tensor | Sequence[float]
_DATAFRAME_PROGRESS_BATCH_SIZE = 256


def _load_pipeline(
    model_name: str,
    *,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> Any:
    """Load a Chronos pipeline while handling old/new dtype kwargs."""
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise ImportError(
            "ChronosExtractor requires chronos-forecasting>=2.1.0.\n"
            "Install it directly or via the project extras, for example:\n"
            "  pip install 'crosslearn[chronos]'"
        ) from exc

    kwargs: dict[str, Any] = {
        "device_map": resolve_device_map(device_map),
        "dtype": dtype,
    }
    try:
        return BaseChronosPipeline.from_pretrained(model_name, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return BaseChronosPipeline.from_pretrained(model_name, **kwargs)


def _as_float_tensor(data: WindowInput) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.detach().to(dtype=torch.float32)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32)


def _pool_embeddings(embeddings: Any, pooling: PoolingMode) -> torch.Tensor:
    def pool_one(embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim == 1:
            return embedding
        token_embeddings = embedding.reshape(-1, embedding.shape[-1])
        if pooling == "last":
            return token_embeddings[-1]
        return token_embeddings.mean(dim=0)

    if isinstance(embeddings, (list, tuple)):
        if not embeddings:
            raise ValueError("Chronos returned an empty embedding batch.")
        return torch.stack([pool_one(_as_float_tensor(item)) for item in embeddings], dim=0)

    tensor = _as_float_tensor(embeddings)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor

    token_embeddings = tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])
    if pooling == "last":
        return token_embeddings[:, -1, :]
    return token_embeddings.mean(dim=1)


def _infer_flat_feature_count(
    flat_dim: int,
    *,
    lookback: int,
    dim_label: str,
) -> int:
    if flat_dim % lookback != 0:
        raise ValueError(f"{dim_label} {flat_dim} is not divisible by lookback={lookback}.")
    return flat_dim // lookback


def _infer_window_layout(
    observation_space: gym.Space,
    *,
    lookback: int | None,
    n_features: int | None,
) -> tuple[int, int]:
    shape = getattr(observation_space, "shape", None)
    if shape is None:
        raise ValueError("ChronosExtractor requires an observation space with a shape.")

    dims = tuple(int(dim) for dim in shape)
    if len(dims) == 2:
        inferred_lookback, inferred_features = dims
        if lookback is not None and lookback != inferred_lookback:
            raise ValueError(
                f"lookback={lookback} does not match observation shape {dims}."
            )
        if n_features is not None and n_features != inferred_features:
            raise ValueError(
                f"n_features={n_features} does not match observation shape {dims}."
            )
        return inferred_lookback, inferred_features

    if len(dims) == 1:
        if lookback is None:
            raise ValueError("ChronosExtractor requires lookback for flat 1D observations.")
        inferred_features = _infer_flat_feature_count(
            dims[0],
            lookback=lookback,
            dim_label="Flat observation dim",
        )
        if n_features is not None and n_features != inferred_features:
            raise ValueError(
                f"n_features={n_features} does not match inferred value "
                f"{inferred_features} from observation shape {dims}."
            )
        return lookback, inferred_features

    raise ValueError(
        "ChronosExtractor expects either a 2D window observation "
        "(lookback, n_features) or a flat 1D observation "
        "(lookback * n_features,)."
    )


def _normalize_window_batch(
    windows: WindowInput,
    *,
    lookback: int | None = None,
    n_features: int | None = None,
) -> tuple[torch.Tensor, int, int]:
    tensor = _as_float_tensor(windows)

    if tensor.ndim == 3:
        inferred_lookback = int(tensor.shape[1])
        inferred_features = int(tensor.shape[2])
        if lookback is not None and inferred_lookback != lookback:
            raise ValueError(
                f"Expected lookback={lookback}, got windows with shape {tuple(tensor.shape)}."
            )
        if n_features is not None and inferred_features != n_features:
            raise ValueError(
                f"Expected n_features={n_features}, got windows with shape {tuple(tensor.shape)}."
            )
        return tensor, inferred_lookback, inferred_features

    if tensor.ndim == 2:
        expected_shape = (
            lookback if lookback is not None else int(tensor.shape[0]),
            n_features if n_features is not None else int(tensor.shape[1]),
        )
        if tuple(tensor.shape) == expected_shape:
            return tensor.unsqueeze(0), int(tensor.shape[0]), int(tensor.shape[1])

        if lookback is None:
            raise ValueError("lookback is required when passing batched flat Chronos windows.")

        flat_dim = int(tensor.shape[1])
        inferred_features = (
            int(n_features)
            if n_features is not None
            else _infer_flat_feature_count(
                flat_dim,
                lookback=lookback,
                dim_label="Flat window dim",
            )
        )
        expected_flat = lookback * inferred_features
        if flat_dim != expected_flat:
            raise ValueError(
                "2D Chronos inputs must be shaped either as a single window "
                "(lookback, n_features) or as batched flat windows "
                "(batch, lookback * n_features)."
            )
        return tensor.reshape(tensor.shape[0], lookback, inferred_features), lookback, inferred_features

    if tensor.ndim == 1:
        if lookback is None:
            raise ValueError("lookback is required when passing a flat Chronos window.")

        inferred_features = (
            int(n_features)
            if n_features is not None
            else _infer_flat_feature_count(
                int(tensor.numel()),
                lookback=lookback,
                dim_label="Flat window dim",
            )
        )
        expected_flat = lookback * inferred_features
        if int(tensor.numel()) != expected_flat:
            raise ValueError(f"Expected flat window size {expected_flat}, got {tensor.numel()}.")
        return tensor.reshape(1, lookback, inferred_features), lookback, inferred_features

    raise ValueError(
        "Chronos windows must be a 1D flat window, 2D single/batched flat window, "
        "or 3D batched time-series tensor."
    )


def _normalize_feature_names(
    feature_names: Sequence[str] | None,
    total_n_features: int,
) -> list[str] | None:
    if feature_names is None:
        return None

    resolved = [str(name) for name in feature_names]
    if len(resolved) != total_n_features:
        raise ValueError(
            f"feature_names has {len(resolved)} entries, but "
            f"{total_n_features} features are present."
        )
    return resolved


def _validate_selection_config(
    *,
    total_n_features: int,
    feature_names: Sequence[str] | None,
    selected_columns: Sequence[str] | None,
    selected_indices: Sequence[int] | None,
) -> tuple[list[int], list[str] | None]:
    if selected_columns is not None and selected_indices is not None:
        raise ValueError("Use either selected_columns or selected_indices, not both.")

    resolved_feature_names = _normalize_feature_names(feature_names, total_n_features)

    if selected_columns is not None:
        if resolved_feature_names is None:
            raise ValueError(
                "selected_columns requires feature_names so column names can be resolved."
            )
        name_to_index = {
            name: idx for idx, name in enumerate(resolved_feature_names)
        }
        missing = [name for name in selected_columns if name not in name_to_index]
        if missing:
            raise ValueError(
                f"Unknown selected_columns {missing}. "
                f"Available columns: {resolved_feature_names}"
            )
        indices = [name_to_index[name] for name in selected_columns]
        return indices, [resolved_feature_names[idx] for idx in indices]

    if selected_indices is None:
        indices = list(range(total_n_features))
    else:
        indices = [int(index) for index in selected_indices]
        for index in indices:
            if index < 0 or index >= total_n_features:
                raise ValueError(
                    f"selected_indices contains {index}, but valid indices are "
                    f"0..{total_n_features - 1}."
                )

    selected_feature_names = None
    if resolved_feature_names is not None:
        selected_feature_names = [resolved_feature_names[idx] for idx in indices]
    return indices, selected_feature_names


def _make_rolling_windows(values: np.ndarray, lookback: int) -> np.ndarray:
    if lookback <= 0:
        raise ValueError("lookback must be greater than 0.")
    if values.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_rows, n_features).")
    if len(values) < lookback:
        raise ValueError(f"Need at least lookback={lookback} rows, got {len(values)}.")

    return np.stack(
        [values[idx : idx + lookback] for idx in range(len(values) - lookback + 1)],
        axis=0,
    )


def _make_dataframe_progress_bar(total_windows: int) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "Chronos dataframe progress bars require tqdm.\n"
            "Install it directly or via the project extras, for example:\n"
            "  pip install 'crosslearn[chronos]'\n"
            "or install tqdm directly:\n"
            "  pip install tqdm"
        ) from exc

    return tqdm(
        total=total_windows,
        unit="window",
        desc="Chronos embeddings",
        dynamic_ncols=True,
    )


def _normalize_frame_bound(frame_bound: Sequence[int]) -> tuple[int, int]:
    if len(frame_bound) != 2:
        raise ValueError("frame_bound must contain exactly two integers: (start, end).")
    return int(frame_bound[0]), int(frame_bound[1])


class ChronosEmbedder:
    """Frozen Chronos utility for online and offline rolling-window embeddings."""

    def __init__(
        self,
        model_name: str = "amazon/chronos-2",
        *,
        pooling: PoolingMode = "mean",
        feature_names: Sequence[str] | None = None,
        selected_columns: Sequence[str] | None = None,
        selected_indices: Sequence[int] | None = None,
        device_map: str = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be either 'mean' or 'last'.")

        self.model_name = model_name
        self.pooling = pooling
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.selected_columns = (
            list(selected_columns) if selected_columns is not None else None
        )
        self.selected_indices = (
            [int(index) for index in selected_indices]
            if selected_indices is not None
            else None
        )
        self.device_map = resolve_device_map(device_map)
        self.dtype = dtype

        self.pipeline = _load_pipeline(model_name, device_map=self.device_map, dtype=dtype)
        self.embedding_dim: int | None = None

    def _resolve_selection(
        self,
        *,
        total_n_features: int,
        feature_names: Sequence[str] | None = None,
    ) -> tuple[list[int], list[str] | None]:
        effective_feature_names = (
            list(feature_names) if feature_names is not None else self.feature_names
        )
        return _validate_selection_config(
            total_n_features=total_n_features,
            feature_names=effective_feature_names,
            selected_columns=self.selected_columns,
            selected_indices=self.selected_indices,
        )

    def embed_windows(
        self,
        windows: WindowInput,
        *,
        lookback: int | None = None,
        n_features: int | None = None,
        feature_names: Sequence[str] | None = None,
        as_tensor: bool = False,
        output_device: torch.device | str | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Embed one or more windows and return one vector per window."""
        normalized_windows, _, inferred_features = _normalize_window_batch(
            windows,
            lookback=lookback,
            n_features=n_features,
        )
        selected_indices, _ = self._resolve_selection(
            total_n_features=inferred_features,
            feature_names=feature_names,
        )
        selected_windows = normalized_windows[..., selected_indices]
        # Chronos stages batches through its own CPU DataLoader/pin-memory path,
        # so embed() must always receive dense CPU tensors even if the model is on CUDA.
        if selected_windows.device.type != "cpu":
            selected_windows = selected_windows.cpu()

        with torch.no_grad():
            result = self.pipeline.embed(selected_windows)

        embeddings = result[0] if isinstance(result, tuple) else result
        pooled = _pool_embeddings(embeddings, self.pooling).to(dtype=torch.float32)
        self.embedding_dim = int(pooled.shape[-1])
        if as_tensor:
            target_device = (
                torch.device(output_device)
                if output_device is not None
                else self.device_map
            )
            if pooled.device != target_device:
                pooled = pooled.to(target_device)
            return pooled
        if pooled.device.type != "cpu":
            pooled = pooled.cpu()
        return pooled.numpy().astype(np.float32, copy=False)

    def _embed_dataframe_windows(
        self,
        windows: np.ndarray,
        *,
        lookback: int,
        feature_names: Sequence[str],
        progress_bar: bool,
    ) -> np.ndarray:
        if not progress_bar:
            return self.embed_windows(
                windows,
                lookback=lookback,
                n_features=len(feature_names),
                feature_names=feature_names,
                as_tensor=False,
            )

        total_windows = int(windows.shape[0])
        progress = _make_dataframe_progress_bar(total_windows)
        embedding_batches: list[np.ndarray] = []
        try:
            for start in range(0, total_windows, _DATAFRAME_PROGRESS_BATCH_SIZE):
                stop = min(start + _DATAFRAME_PROGRESS_BATCH_SIZE, total_windows)
                embedding_batches.append(
                    self.embed_windows(
                        windows[start:stop],
                        lookback=lookback,
                        n_features=len(feature_names),
                        feature_names=feature_names,
                        as_tensor=False,
                    )
                )
                progress.update(stop - start)
        finally:
            progress.close()

        return np.concatenate(embedding_batches, axis=0)

    def transform_dataframe(
        self,
        df: Any,
        *,
        lookback: int,
        columns: Sequence[str] | None = None,
        output_prefix: str = "chronos_",
        progress_bar: bool = False,
    ) -> Any:
        """Append aligned Chronos embedding columns to a dataframe.

        Set ``progress_bar=True`` to batch the offline embedding pass and show a
        ``tqdm`` progress bar.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Chronos dataframe helpers require pandas.\n"
                "Install it directly, for example:\n"
                "  pip install pandas"
            ) from exc

        if not isinstance(df, pd.DataFrame):
            raise TypeError("transform_dataframe expects a pandas.DataFrame.")

        if columns is not None:
            source_columns = [str(column) for column in columns]
        elif self.feature_names is not None:
            source_columns = list(self.feature_names)
        else:
            source_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not source_columns:
            raise ValueError(
                "No numeric dataframe columns are available for Chronos embeddings."
            )

        missing = [column for column in source_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing dataframe columns for Chronos embeddings: {missing}")

        windows = _make_rolling_windows(
            df.loc[:, source_columns].to_numpy(dtype=np.float32, copy=True),
            lookback=lookback,
        )
        embeddings = self._embed_dataframe_windows(
            windows,
            lookback=lookback,
            feature_names=source_columns,
            progress_bar=progress_bar,
        )

        embedding_columns = [
            f"{output_prefix}{index}" for index in range(int(embeddings.shape[1]))
        ]
        embedding_frame = pd.DataFrame(
            data=np.nan,
            index=df.index,
            columns=embedding_columns,
            dtype=np.float32,
        )
        embedding_frame.iloc[lookback - 1 :] = embeddings
        return pd.concat([df.copy(), embedding_frame], axis=1)

    def __repr__(self) -> str:
        return (
            f"ChronosEmbedder(model_name={self.model_name!r}, "
            f"pooling={self.pooling!r}, "
            f"selected_columns={self.selected_columns}, "
            f"selected_indices={self.selected_indices})"
        )


def build_offline_bundle(
    df: Any,
    *,
    lookback: int,
    frame_bound: Sequence[int],
    feature_columns: Sequence[str],
    selected_columns: Sequence[str] | None = None,
    selected_indices: Sequence[int] | None = None,
    output_prefix: str = "chronos_",
    progress_bar: bool = False,
    model_name: str = "amazon/chronos-2",
    pooling: PoolingMode = "mean",
    device_map: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Build an aligned offline Chronos bundle for dataframe-backed environments."""
    if lookback <= 0:
        raise ValueError("lookback must be greater than 0.")

    frame_start, frame_end = _normalize_frame_bound(frame_bound)
    if frame_start < lookback:
        raise ValueError(
            f"frame_bound[0] must be at least lookback={lookback}, got {frame_start}."
        )
    if frame_start >= frame_end:
        raise ValueError(
            f"frame_bound must satisfy frame_bound[0] < frame_bound[1], got {frame_bound}."
        )
    if frame_end > len(df):
        raise ValueError(
            f"frame_bound[1] must be <= len(df)={len(df)}, got {frame_end}."
        )

    resolved_feature_columns = [str(column) for column in feature_columns]
    if not resolved_feature_columns:
        raise ValueError("feature_columns must contain at least one column name.")

    history = df.iloc[frame_start - lookback : frame_end].reset_index(drop=True).copy()
    embedder = ChronosEmbedder(
        model_name=model_name,
        pooling=pooling,
        feature_names=resolved_feature_columns,
        selected_columns=selected_columns,
        selected_indices=selected_indices,
        device_map=device_map,
        dtype=dtype,
    )
    transformed = embedder.transform_dataframe(
        history,
        lookback=lookback,
        columns=resolved_feature_columns,
        output_prefix=output_prefix,
        progress_bar=progress_bar,
    )
    embedding_columns = [
        column for column in transformed.columns if str(column).startswith(output_prefix)
    ]
    trimmed_df = transformed.iloc[lookback - 1 :].reset_index(drop=True)
    embedding_frame = trimmed_df.loc[:, embedding_columns].copy()
    return {
        "df": trimmed_df,
        "embedding_frame": embedding_frame,
    }


class ChronosExtractor(BaseFeaturesExtractor):
    """Frozen Chronos-2 extractor for raw rolling windows or flat legacy inputs.

    This extractor is designed as a reusable backbone for Gymnasium and SB3
    policies rather than a reproduction of ChronosRL.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int | None = None,
        model_name: str = "amazon/chronos-2",
        lookback: int | None = None,
        n_features: int | None = None,
        freeze: bool = True,
        pooling: PoolingMode = "mean",
        feature_names: Sequence[str] | None = None,
        selected_columns: Sequence[str] | None = None,
        selected_indices: Sequence[int] | None = None,
        device_map: str = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(observation_space, features_dim or 1)

        if not freeze:
            raise ValueError(
                "ChronosExtractor uses fixed Chronos embeddings. Keep freeze=True."
            )

        self.lookback, self.n_features = _infer_window_layout(
            observation_space,
            lookback=lookback,
            n_features=n_features,
        )
        self.model_name = model_name
        self.pooling = pooling
        self.feature_names = list(feature_names) if feature_names is not None else None

        self.embedder = ChronosEmbedder(
            model_name=model_name,
            pooling=pooling,
            feature_names=self.feature_names,
            selected_columns=selected_columns,
            selected_indices=selected_indices,
            device_map=device_map,
            dtype=dtype,
        )
        self.selected_indices, self.selected_feature_names = self.embedder._resolve_selection(
            total_n_features=self.n_features,
            feature_names=self.feature_names,
        )

        example_features = self.embedder.embed_windows(
            torch.zeros((1, self.lookback, self.n_features), dtype=torch.float32),
            lookback=self.lookback,
            n_features=self.n_features,
            feature_names=self.feature_names,
            as_tensor=True,
        )
        self.embedding_dim = int(example_features.shape[-1])
        resolved_features_dim = (
            self.embedding_dim if features_dim is None else int(features_dim)
        )
        self._features_dim = resolved_features_dim

        if resolved_features_dim == self.embedding_dim:
            self.projection: nn.Module = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(self.embedding_dim, resolved_features_dim),
                nn.ReLU(),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        embedded = self.embedder.embed_windows(
            observations,
            lookback=self.lookback,
            n_features=self.n_features,
            feature_names=self.feature_names,
            as_tensor=True,
            output_device=observations.device,
        )
        return self.projection(embedded)

    def __repr__(self) -> str:
        return (
            f"ChronosExtractor(model_name={self.model_name!r}, "
            f"lookback={self.lookback}, "
            f"n_features={self.n_features}, "
            f"selected_indices={self.selected_indices}, "
            f"embedding_dim={self.embedding_dim}, "
            f"features_dim={self.features_dim})"
        )

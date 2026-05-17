"""Chronos-backed extractors and embedders for rolling time-series windows.

The utilities in this module are original package components for reusable
Chronos-backed RL features. They are not an implementation of ChronosRL
(Lima, Oliveira, and Zanchettin, 2025), though that paper is relevant adjacent
inspiration for Chronos-based reinforcement learning on market data.
"""

from __future__ import annotations

import inspect
import warnings
from collections import OrderedDict
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
_DEFAULT_ONLINE_CACHE_SIZE = 16_384


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


def _stack_embedding_items(embeddings: Any) -> torch.Tensor:
    if isinstance(embeddings, (list, tuple)):
        if not embeddings:
            raise ValueError("Chronos returned an empty embedding batch.")
        return torch.stack([_as_float_tensor(item) for item in embeddings], dim=0)
    return _as_float_tensor(embeddings)


def _infer_model_device(model: Any, fallback: str | torch.device) -> torch.device:
    device = getattr(model, "device", None)
    if device is None:
        try:
            first_parameter = next(model.parameters())
        except (AttributeError, StopIteration):
            return torch.device(fallback)
        return first_parameter.device
    return torch.device(device)


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
    """Normalize windows to Chronos layout ``(batch, n_features, lookback)``."""
    tensor = _as_float_tensor(windows)

    if tensor.ndim == 3:
        if lookback is not None and n_features is not None:
            if tuple(tensor.shape[1:]) == (lookback, n_features):
                return tensor.transpose(1, 2), lookback, n_features
            if tuple(tensor.shape[1:]) == (n_features, lookback):
                return tensor, lookback, n_features
            raise ValueError(
                f"Expected batched Chronos windows shaped either "
                f"(batch, lookback={lookback}, n_features={n_features}) or "
                f"(batch, n_features={n_features}, lookback={lookback}), got "
                f"{tuple(tensor.shape)}."
            )

        if (
            lookback is not None
            and tensor.shape[2] == lookback
            and tensor.shape[1] != lookback
        ):
            inferred_features = int(tensor.shape[1])
            return tensor, lookback, inferred_features

        if (
            n_features is not None
            and tensor.shape[1] == n_features
            and tensor.shape[2] != n_features
        ):
            inferred_lookback = int(tensor.shape[2])
            return tensor, inferred_lookback, n_features

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
        return tensor.transpose(1, 2), inferred_lookback, inferred_features

    if tensor.ndim == 2:
        expected_shape = (
            lookback if lookback is not None else int(tensor.shape[0]),
            n_features if n_features is not None else int(tensor.shape[1]),
        )
        if tuple(tensor.shape) == expected_shape:
            return tensor.transpose(0, 1).unsqueeze(0), int(tensor.shape[0]), int(tensor.shape[1])
        if (
            lookback is not None
            and n_features is not None
            and tuple(tensor.shape) == (n_features, lookback)
        ):
            return tensor.unsqueeze(0), lookback, n_features

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
        batched_windows = tensor.reshape(tensor.shape[0], lookback, inferred_features)
        return batched_windows.transpose(1, 2), lookback, inferred_features

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
        canonical = tensor.reshape(1, lookback, inferred_features).transpose(1, 2)
        return canonical, lookback, inferred_features

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


def _resolve_dataframe_feature_names(
    df: Any,
    *,
    feature_names: Sequence[str] | None = None,
    context: str = "Chronos dataframe input",
) -> list[str]:
    if feature_names is not None:
        resolved = [str(name) for name in feature_names]
    else:
        resolved = df.select_dtypes(include=[np.number]).columns.tolist()

    if not resolved:
        raise ValueError(f"{context} requires at least one numeric feature column.")

    missing = [column for column in resolved if column not in df.columns]
    if missing:
        raise ValueError(f"Missing dataframe columns for {context}: {missing}")

    return resolved


class ChronosEmbedder:
    """Frozen Chronos wrapper for online and offline rolling-window embeddings.

    This utility loads a pretrained Chronos pipeline once and exposes two
    user-facing workflows:

    - ``embed_windows(...)`` embeds one or more rolling windows directly
    - ``transform_dataframe(...)`` appends aligned ``chronos_*`` columns to a
      full dataframe

    "Frozen" means the Chronos model is used in inference mode only. The
    package does not fine-tune the Chronos weights.

    Feature selection happens before the batch is sent to Chronos. Provide
    ``feature_names`` to name the raw feature axis, then optionally keep only a
    subset with ``selected_columns`` or ``selected_indices``.

    Args:
        model_name: Hugging Face or Chronos model identifier to load.
            Default: ``"amazon/chronos-2"``.
        pooling: How token-level Chronos embeddings are pooled into one vector
            per input window. ``"mean"`` averages token embeddings and
            ``"last"`` keeps the last token embedding.
        feature_names: Optional names for the raw feature axis. These are used
            to resolve ``selected_columns`` and to document the expected feature
            ordering.
        selected_columns: Optional subset of ``feature_names`` to embed by
            name. Mutually exclusive with ``selected_indices``.
        selected_indices: Optional subset of feature positions to embed by
            index. Mutually exclusive with ``selected_columns``.
        device_map: Target device for the Chronos model, for example
            ``"auto"``, ``"cpu"``, or ``"cuda"``.
        dtype: Torch dtype used when loading the Chronos pipeline.
        embed_batch_size: Batch size forwarded to Chronos embedding APIs that
            support it.
        cache_size: Optional LRU cache size for repeated selected windows.

    Example::

        embedder = ChronosEmbedder(
            feature_names=["open", "high", "low", "close"],
            pooling="mean",
        )
        window = np.zeros((32, 4), dtype=np.float32)
        embedding = embedder.embed_windows(window, lookback=32, n_features=4)
        embedding.shape  # (1, embedding_dim)
    """

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
        embed_batch_size: int = 256,
        cache_size: int | None = None,
    ) -> None:
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be either 'mean' or 'last'.")
        if embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be greater than 0.")
        if cache_size is not None and cache_size < 0:
            raise ValueError("cache_size must be non-negative or None.")

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
        self.embed_batch_size = int(embed_batch_size)
        self.cache_size = None if cache_size is None else int(cache_size)

        self.pipeline = _load_pipeline(model_name, device_map=self.device_map, dtype=dtype)
        model = getattr(self.pipeline, "model", None)
        if callable(getattr(model, "eval", None)):
            model.eval()
        self.embedding_dim: int | None = None
        self._embedding_cache: OrderedDict[tuple[tuple[int, ...], bytes], torch.Tensor] = (
            OrderedDict()
        )
        self.last_embedding_path: str | None = None
        self.last_fallback_reason: str | None = None
        self._warned_direct_fallback = False

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

    def clear_cache(self) -> None:
        """Clear cached Chronos embeddings held by this embedder."""
        self._embedding_cache.clear()

    def _cache_enabled(self) -> bool:
        return self.cache_size is not None and self.cache_size > 0

    def _cache_key(self, window: torch.Tensor) -> tuple[tuple[int, ...], bytes]:
        contiguous = window.detach().to(device="cpu", dtype=torch.float32).contiguous()
        return tuple(contiguous.shape), contiguous.numpy().tobytes()

    def _get_cached_embedding(
        self,
        key: tuple[tuple[int, ...], bytes],
    ) -> torch.Tensor | None:
        cached = self._embedding_cache.get(key)
        if cached is None:
            return None
        self._embedding_cache.move_to_end(key)
        return cached.clone()

    def _store_cached_embedding(
        self,
        key: tuple[tuple[int, ...], bytes],
        embedding: torch.Tensor,
    ) -> None:
        if not self._cache_enabled():
            return
        self._embedding_cache[key] = embedding.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).clone()
        self._embedding_cache.move_to_end(key)
        while self.cache_size is not None and len(self._embedding_cache) > self.cache_size:
            self._embedding_cache.popitem(last=False)

    def _is_amazon_chronos_model(self) -> bool:
        return self.model_name.lower().startswith("amazon/chronos")

    def _is_chronos2_pipeline(self) -> bool:
        if "chronos-2" in self.model_name.lower():
            return True
        pipeline_name = type(self.pipeline).__name__.lower()
        if "chronos2" in pipeline_name or "chronos_2" in pipeline_name:
            return True
        model = getattr(self.pipeline, "model", None)
        model_name = type(model).__name__.lower()
        return "chronos2" in model_name or "chronos_2" in model_name

    def _is_chronos_bolt_pipeline(self) -> bool:
        if "chronos-bolt" in self.model_name.lower():
            return True
        pipeline_name = type(self.pipeline).__name__.lower()
        model = getattr(self.pipeline, "model", None)
        model_name = type(model).__name__.lower()
        return "bolt" in pipeline_name or "bolt" in model_name

    def _is_chronos_t5_pipeline(self) -> bool:
        if "chronos-t5" in self.model_name.lower():
            return True
        tokenizer = getattr(self.pipeline, "tokenizer", None)
        if not callable(getattr(tokenizer, "context_input_transform", None)):
            return False
        model = getattr(self.pipeline, "model", None)
        return callable(getattr(model, "encode", None))

    def _call_pipeline_embed(self, context: torch.Tensor) -> Any:
        embed = self.pipeline.embed
        try:
            signature = inspect.signature(embed)
        except (TypeError, ValueError):
            signature = None

        if signature is not None and "batch_size" in signature.parameters:
            return embed(context, batch_size=self.embed_batch_size)
        return embed(context)

    def _embed_chronos2_direct(self, selected_windows: torch.Tensor) -> torch.Tensor:
        model = self.pipeline.model
        batch_size, n_variates, lookback = selected_windows.shape
        model_context_length = getattr(model.chronos_config, "context_length", None)
        if model_context_length is not None and lookback > int(model_context_length):
            lookback = int(model_context_length)
            selected_windows = selected_windows[..., -lookback:]

        model_device = _infer_model_device(model, self.device_map)
        context = selected_windows.reshape(batch_size * n_variates, lookback).to(
            device=model_device,
            dtype=torch.float32,
        )
        group_ids = torch.arange(batch_size, device=model_device).repeat_interleave(
            n_variates
        )

        with torch.inference_mode():
            result = model.encode(context=context, group_ids=group_ids)

        encoder_outputs = result[0] if isinstance(result, tuple) else result
        hidden_states = (
            encoder_outputs[0]
            if isinstance(encoder_outputs, (tuple, list))
            else getattr(encoder_outputs, "last_hidden_state", encoder_outputs)
        )
        hidden_states = _as_float_tensor(hidden_states)
        grouped = hidden_states.reshape(batch_size, n_variates, *hidden_states.shape[1:])
        pooled = _pool_embeddings(grouped, self.pooling).to(dtype=torch.float32)
        if pooled.device.type != "cpu":
            pooled = pooled.cpu()
        return pooled

    def _embed_chronos_bolt_direct(self, selected_windows: torch.Tensor) -> torch.Tensor:
        model = self.pipeline.model
        batch_size, n_variates, lookback = selected_windows.shape
        model_device = _infer_model_device(model, self.device_map)
        context = selected_windows.reshape(batch_size * n_variates, lookback).to(
            device=model_device,
            dtype=torch.float32,
        )
        mask = torch.isnan(context).logical_not()

        with torch.inference_mode():
            result = model.encode(context=context, mask=mask)

        hidden_states = result[0] if isinstance(result, tuple) else result
        hidden_states = _as_float_tensor(hidden_states)
        grouped = hidden_states.reshape(batch_size, n_variates, *hidden_states.shape[1:])
        pooled = _pool_embeddings(grouped, self.pooling).to(dtype=torch.float32)
        if pooled.device.type != "cpu":
            pooled = pooled.cpu()
        return pooled

    def _embed_chronos_t5_direct(self, selected_windows: torch.Tensor) -> torch.Tensor:
        tokenizer = self.pipeline.tokenizer
        model = self.pipeline.model
        batch_size, n_variates, lookback = selected_windows.shape
        context = selected_windows.reshape(batch_size * n_variates, lookback)

        token_ids, attention_mask, _ = tokenizer.context_input_transform(context)
        model_device = _infer_model_device(model, self.device_map)
        with torch.inference_mode():
            hidden_states = model.encode(
                input_ids=token_ids.to(model_device),
                attention_mask=attention_mask.to(model_device),
            )

        hidden_states = _as_float_tensor(hidden_states)
        grouped = hidden_states.reshape(batch_size, n_variates, *hidden_states.shape[1:])
        pooled = _pool_embeddings(grouped, self.pooling).to(dtype=torch.float32)
        if pooled.device.type != "cpu":
            pooled = pooled.cpu()
        return pooled

    def _embed_direct_amazon(self, selected_windows: torch.Tensor) -> torch.Tensor | None:
        if self._is_chronos2_pipeline():
            self.last_embedding_path = "chronos2_direct"
            return self._embed_chronos2_direct(selected_windows)
        if self._is_chronos_bolt_pipeline():
            self.last_embedding_path = "chronos_bolt_direct"
            return self._embed_chronos_bolt_direct(selected_windows)
        if self._is_chronos_t5_pipeline():
            self.last_embedding_path = "chronos_t5_direct"
            return self._embed_chronos_t5_direct(selected_windows)
        return None

    def _warn_direct_fallback(self, exc: BaseException | str) -> None:
        self.last_fallback_reason = str(exc)
        if self._warned_direct_fallback:
            return
        self._warned_direct_fallback = True
        warnings.warn(
            "Direct Chronos model encoding failed or was unavailable; falling back "
            "to pipeline.embed(...), which may be slower. Reason: "
            f"{self.last_fallback_reason}",
            RuntimeWarning,
            stacklevel=3,
        )

    def _embed_with_pipeline(self, selected_windows: torch.Tensor) -> torch.Tensor:
        if self._is_chronos2_pipeline():
            result = self._call_pipeline_embed(selected_windows)
            embeddings = result[0] if isinstance(result, tuple) else result
            pooled = _pool_embeddings(embeddings, self.pooling).to(dtype=torch.float32)
            if pooled.device.type != "cpu":
                pooled = pooled.cpu()
            return pooled

        batch_size, n_variates, lookback = selected_windows.shape
        flat_context = selected_windows.reshape(batch_size * n_variates, lookback)
        result = self._call_pipeline_embed(flat_context)
        embeddings = result[0] if isinstance(result, tuple) else result
        flat_embeddings = _stack_embedding_items(embeddings).to(dtype=torch.float32)
        grouped = flat_embeddings.reshape(batch_size, n_variates, *flat_embeddings.shape[1:])
        pooled = _pool_embeddings(grouped, self.pooling).to(dtype=torch.float32)
        if pooled.device.type != "cpu":
            pooled = pooled.cpu()
        return pooled

    def _embed_selected_windows_direct_or_fallback(
        self,
        selected_windows: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_amazon_chronos_model():
            try:
                direct_embeddings = self._embed_direct_amazon(selected_windows)
                if direct_embeddings is not None:
                    self.last_fallback_reason = None
                    return direct_embeddings
                self._warn_direct_fallback(
                    f"no direct adapter matched {self.model_name!r}"
                )
            except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
                self._warn_direct_fallback(exc)

        # Chronos pipeline.embed implementations stage batches through CPU
        # validation/DataLoader paths, so the fallback always receives CPU tensors.
        self.last_embedding_path = "pipeline_embed"
        return self._embed_with_pipeline(selected_windows)

    def _embed_selected_windows_uncached(
        self,
        selected_windows: torch.Tensor,
    ) -> torch.Tensor:
        n_variates = max(1, int(selected_windows.shape[1]))
        max_windows_per_chunk = max(1, self.embed_batch_size // n_variates)
        if int(selected_windows.shape[0]) <= max_windows_per_chunk:
            return self._embed_selected_windows_direct_or_fallback(selected_windows)

        embedding_batches: list[torch.Tensor] = []
        for start in range(0, int(selected_windows.shape[0]), max_windows_per_chunk):
            stop = min(start + max_windows_per_chunk, int(selected_windows.shape[0]))
            embedding_batches.append(
                self._embed_selected_windows_direct_or_fallback(
                    selected_windows[start:stop]
                )
            )
        return torch.cat(embedding_batches, dim=0)

    def _embed_selected_windows(
        self,
        selected_windows: torch.Tensor,
    ) -> torch.Tensor:
        selected_windows = selected_windows.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous()

        if not self._cache_enabled():
            return self._embed_selected_windows_uncached(selected_windows)

        outputs: list[torch.Tensor | None] = [None] * int(selected_windows.shape[0])
        pending_keys: list[tuple[tuple[int, ...], bytes]] = []
        pending_windows: list[torch.Tensor] = []
        pending_positions: dict[tuple[tuple[int, ...], bytes], list[int]] = {}

        for index, window in enumerate(selected_windows):
            key = self._cache_key(window)
            cached = self._get_cached_embedding(key)
            if cached is not None:
                outputs[index] = cached
                continue
            if key not in pending_positions:
                pending_keys.append(key)
                pending_windows.append(window)
                pending_positions[key] = []
            pending_positions[key].append(index)

        if pending_windows:
            miss_batch = torch.stack(pending_windows, dim=0)
            miss_embeddings = self._embed_selected_windows_uncached(miss_batch)
            for key, embedding in zip(pending_keys, miss_embeddings, strict=True):
                cpu_embedding = embedding.detach().to(device="cpu", dtype=torch.float32)
                self._store_cached_embedding(key, cpu_embedding)
                for position in pending_positions[key]:
                    outputs[position] = cpu_embedding.clone()

        filled_outputs: list[torch.Tensor] = []
        for output in outputs:
            if output is None:
                raise RuntimeError("Chronos embedding cache failed to populate all outputs.")
            filled_outputs.append(output)
        return torch.stack(filled_outputs, dim=0)

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
        """Embed one or more rolling windows and return one vector per window.

        Accepted input layouts are:

        - flat 1D window: ``(lookback * n_features,)``
        - 2D single window: ``(lookback, n_features)``
        - 2D batch of flat windows: ``(batch, lookback * n_features)``
        - 3D batch of windows: ``(batch, lookback, n_features)`` or
          ``(batch, n_features, lookback)``

        Args:
            windows: One window or a batch of windows in any supported layout.
            lookback: Number of timesteps per window. Required when the input is
                flat or when the layout cannot be inferred unambiguously.
            n_features: Number of raw features per timestep. Optional when the
                layout already implies it.
            feature_names: Optional feature names for this call. When omitted,
                ``self.feature_names`` is used.
            as_tensor: If ``True``, return a ``torch.Tensor``. Otherwise return
                a CPU ``numpy.ndarray``.
            output_device: Target device for tensor outputs when
                ``as_tensor=True``. Defaults to the embedder device.

        Returns:
            One pooled Chronos embedding per input window with shape
            ``(batch, embedding_dim)`` and dtype ``float32``. NumPy outputs are
            always returned on CPU. Tensor outputs are moved to
            ``output_device`` when provided.

        Raises:
            ValueError: If the window shape is incompatible with
                ``lookback``/``n_features`` or if feature selection cannot be
                resolved.

        Example::

            batch = np.zeros((8, 32, 4), dtype=np.float32)
            embeddings = embedder.embed_windows(
                batch,
                lookback=32,
                n_features=4,
            )
            embeddings.shape  # (8, embedding_dim)
        """
        normalized_windows, _, inferred_features = _normalize_window_batch(
            windows,
            lookback=lookback,
            n_features=n_features,
        )
        selected_indices, _ = self._resolve_selection(
            total_n_features=inferred_features,
            feature_names=feature_names,
        )
        selected_windows = normalized_windows[:, selected_indices, :]
        pooled = self._embed_selected_windows(selected_windows).to(dtype=torch.float32)
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

        The returned dataframe has the same length and index as ``df``. The
        first ``lookback - 1`` rows in the new embedding columns are ``NaN``
        because there is not yet enough history to form a complete rolling
        window.

        When ``columns`` is omitted, the embedder uses ``self.feature_names`` if
        available, otherwise all numeric dataframe columns.

        Args:
            df: Source pandas dataframe in chronological order.
            lookback: Number of rows per rolling window.
            columns: Source dataframe columns to read before any optional
                ``selected_columns`` or ``selected_indices`` filtering is
                applied.
            output_prefix: Prefix for the appended embedding columns.
            progress_bar: If ``True``, batch the offline embedding pass and show
                a ``tqdm`` progress bar.

        Returns:
            A copy of ``df`` with appended ``{output_prefix}*`` embedding
            columns aligned to the original index.

        Raises:
            TypeError: If ``df`` is not a pandas dataframe.
            ValueError: If the source columns are missing or no usable numeric
                columns can be resolved.

        Example::

            embedded = embedder.transform_dataframe(
                df,
                lookback=32,
                columns=["open", "high", "low", "close"],
            )
            embedded.filter(like="chronos_").iloc[:31].isna().all().all()
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


def embed_dataframe(
    df: Any,
    *,
    lookback: int,
    frame_bound: Sequence[int],
    feature_names: Sequence[str] | None = None,
    selected_columns: Sequence[str] | None = None,
    selected_indices: Sequence[int] | None = None,
    output_prefix: str = "chronos_",
    progress_bar: bool = False,
    model_name: str = "amazon/chronos-2",
    pooling: PoolingMode = "mean",
    device_map: str = "auto",
    dtype: torch.dtype = torch.float32,
    embed_batch_size: int = 256,
    cache_size: int | None = None,
    drop_feature_names: bool = False,
) -> Any:
    """Build a trimmed offline Chronos dataframe for dataframe-backed envs.

    This helper mirrors the rolling-window alignment used by dataframe-backed
    trading environments. It:

    1. slices ``df.iloc[frame_bound[0] - lookback : frame_bound[1]]``
    2. appends aligned Chronos embedding columns over that slice
    3. drops the leading ``lookback - 1`` warmup rows
    4. resets the index on the trimmed result

    The returned rows therefore align with the windowed observation stream, not
    the original raw dataframe index. In particular, the first returned row
    corresponds to the first complete rolling window ending immediately before
    the environment's first agent-visible index.

    Args:
        df: Source pandas dataframe in chronological order.
        lookback: Number of rows per rolling window.
        frame_bound: Two-element slice describing the environment span to
            support. The helper uses the preceding ``lookback`` rows as history.
        feature_names: Raw dataframe columns to read before optional feature
            selection is applied inside ``ChronosEmbedder``. When omitted, all
            numeric dataframe columns are used.
        selected_columns: Optional subset of ``feature_names`` to embed by
            name.
        selected_indices: Optional subset of ``feature_names`` to embed by
            position.
        output_prefix: Prefix for appended embedding columns.
        progress_bar: If ``True``, show the offline embedding progress.
        model_name: Chronos model identifier.
        pooling: Token pooling mode forwarded to ``ChronosEmbedder``.
        device_map: Target device for the Chronos model.
        dtype: Torch dtype used when loading Chronos.
        embed_batch_size: Batch size forwarded to Chronos embedding APIs that
            support it.
        cache_size: Optional LRU cache size for repeated windows. Disabled by
            default for offline embedding.
        drop_feature_names: If ``True``, drop the resolved source features
            from the returned dataframe after embedding.

    Returns:
        A trimmed, reindexed dataframe containing one row per aligned
        rolling-window observation, with appended ``chronos_*`` columns by
        default.

    Raises:
        ValueError: If ``lookback`` or ``frame_bound`` are invalid, or if no
            numeric features can be resolved.

    Example::

        embedded = embed_dataframe(
            df,
            lookback=32,
            frame_bound=(32, len(df)),
            feature_names=["open", "high", "low", "close", "volume"],
        )
        embedded.filter(like="chronos_").head()
    """
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

    resolved_feature_names = _resolve_dataframe_feature_names(
        df,
        feature_names=feature_names,
        context="Chronos embeddings",
    )

    history = df.iloc[frame_start - lookback : frame_end].reset_index(drop=True).copy()
    embedder = ChronosEmbedder(
        model_name=model_name,
        pooling=pooling,
        feature_names=resolved_feature_names,
        selected_columns=selected_columns,
        selected_indices=selected_indices,
        device_map=device_map,
        dtype=dtype,
        embed_batch_size=embed_batch_size,
        cache_size=cache_size,
    )
    transformed = embedder.transform_dataframe(
        history,
        lookback=lookback,
        columns=resolved_feature_names,
        output_prefix=output_prefix,
        progress_bar=progress_bar,
    )
    trimmed = transformed.iloc[lookback - 1 :].reset_index(drop=True)
    if drop_feature_names:
        trimmed = trimmed.drop(columns=resolved_feature_names)
    return trimmed


class ChronosExtractor(BaseFeaturesExtractor):
    """Frozen Chronos-2 extractor for raw rolling windows or flat legacy inputs.

    This extractor is designed as a reusable backbone for Gymnasium and SB3
    policies rather than a reproduction of ChronosRL.

    Each observation must represent exactly one rolling window, either as a 2D
    matrix ``(lookback, n_features)`` or as a flat legacy vector
    ``(lookback * n_features,)``. During policy execution a leading batch
    dimension is added automatically, so typical model inputs look like
    ``(batch, lookback, n_features)`` or ``(batch, lookback * n_features)``.

    The extractor first obtains a frozen Chronos embedding for each
    observation. If ``features_dim`` is omitted, that embedding is returned
    directly. If ``features_dim`` differs from the raw Chronos embedding width,
    the extractor adds a learned ``Linear + ReLU`` projection so downstream
    policies still receive the requested feature dimension.

    Use this extractor when the environment already emits raw rolling windows
    and you only need fixed Chronos features. If you need online walk-forward
    PCA before the policy sees each observation, wrap the environment with
    ``WalkForwardChronosPCAWrapper`` instead.

    Args:
        observation_space: Window-shaped observation space. Supported layouts
            are ``(lookback, n_features)`` and ``(lookback * n_features,)``.
        features_dim: Output width seen by the policy. When omitted, the raw
            Chronos embedding width is used.
        model_name: Chronos model identifier. Default: ``"amazon/chronos-2"``.
        lookback: Required for flat 1D observation spaces. Optional for 2D
            observation spaces where it can be inferred.
        n_features: Required for flat 1D observation spaces. Optional for 2D
            observation spaces where it can be inferred.
        freeze: Must remain ``True``. Trainable Chronos fine-tuning is not
            supported in this extractor.
        pooling: Token pooling mode used to turn Chronos token embeddings into
            one vector per observation.
        feature_names: Optional names for the raw feature axis.
        selected_columns: Optional subset of ``feature_names`` to embed by
            name.
        selected_indices: Optional subset of feature positions to embed by
            index.
        device_map: Target device for the Chronos model.
        dtype: Torch dtype used when loading Chronos.
        cache_size: Optional LRU cache size for repeated selected windows.
            Defaults to ``16384`` for the online extractor.

    Example::

        extractor = ChronosExtractor(
            env.observation_space,
            lookback=32,
            n_features=5,
        )
        obs = torch.zeros(8, 32, 5)
        out = extractor(obs)
        out.shape  # (8, extractor.features_dim)
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
        cache_size: int | None = _DEFAULT_ONLINE_CACHE_SIZE,
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
            cache_size=cache_size,
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
        self.embedder.clear_cache()
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
        """Encode a batch of rolling windows into Chronos feature vectors.

        Args:
            observations: Batch of raw windows in either window layout
                ``(batch, lookback, n_features)`` or flat layout
                ``(batch, lookback * n_features)``.

        Returns:
            A tensor of shape ``(batch, features_dim)`` on the same device as
            ``observations``.
        """
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

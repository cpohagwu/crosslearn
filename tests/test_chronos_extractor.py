from __future__ import annotations

import sys
import types
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import pytest
import torch

from crosslearn import REINFORCE
from crosslearn.extractors import (
    embed_dataframe as exported_embed_dataframe,
)
import crosslearn.extractors.chronos as chronos_module
from crosslearn.extractors.base import BaseFeaturesExtractor
from crosslearn.extractors.chronos import (
    ChronosEmbedder,
    ChronosExtractor,
    embed_dataframe,
)


def _make_chronos_dataframe():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Close": [2.0, 3.0, 4.0, 5.0, 6.0],
            "Volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )


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


class _TinyChronosEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, 5),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        return np.zeros((4, 5), dtype=np.float32), {}

    def step(self, action: int):
        return np.zeros((4, 5), dtype=np.float32), 0.0, True, False, {}


class _StrictCpuInputChronosPipeline:
    def __init__(self) -> None:
        self.device = torch.device("meta")
        self.calls: list[torch.Tensor] = []

    def embed(self, context):
        context_t = torch.as_tensor(context, dtype=torch.float32)
        if context_t.device.type != "cpu":
            raise RuntimeError("Chronos embed expected a CPU tensor input.")
        self.calls.append(context_t.clone())

        if context_t.ndim == 2:
            context_t = context_t.unsqueeze(0)

        summary = torch.stack(
            [
                context_t.mean(dim=(1, 2)),
                context_t[:, :, 0].mean(dim=1),
                context_t[:, :, -1].mean(dim=1),
                context_t[:, -1, :].mean(dim=1),
            ],
            dim=-1,
        )
        embeddings = [
            torch.stack([summary[index], summary[index] + 5.0], dim=0)
            for index in range(summary.shape[0])
        ]
        return embeddings, {"dummy": True}


def test_base_features_extractor_requires_forward_implementation() -> None:
    class MissingForwardExtractor(BaseFeaturesExtractor):
        pass

    with pytest.raises(TypeError, match="abstract"):
        MissingForwardExtractor(
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            4,
        )


def test_chronos_extractor_accepts_anytrading_windows_and_selected_columns(
    fake_chronos,
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 5),
        dtype=np.float32,
    )
    extractor = ChronosExtractor(
        observation_space,
        model_name="amazon/chronos-2",
        feature_names=["Open", "High", "Low", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )

    batch = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
    features = extractor(batch)

    assert features.shape == (2, 4)
    assert extractor.lookback == 4
    assert extractor.n_features == 5
    assert extractor.selected_indices == [3, 4]
    assert fake_chronos.last_model_name == "amazon/chronos-2"
    assert fake_chronos.last_kwargs == {"device_map": "cpu", "dtype": torch.float32}
    assert fake_chronos.last_pipeline is not None
    assert fake_chronos.last_pipeline.calls[-1].shape == (2, 4, 2)


def test_reinforce_aligns_default_chronos_device_map_with_cuda_agent(
    fake_chronos,
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *args, **kwargs: self)
    original_tensor_to = torch.Tensor.to

    def _tensor_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target is not None and torch.device(target).type == "cuda":
            return self
        return original_tensor_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _tensor_to, raising=False)

    agent = REINFORCE(
        lambda: _TinyChronosEnv(),
        n_envs=1,
        n_steps=1,
        features_extractor_class=ChronosExtractor,
        features_extractor_kwargs={
            "feature_names": ["Open", "High", "Low", "Close", "Volume"],
            "selected_columns": ["Close", "Volume"],
        },
        verbose=0,
    )

    assert agent.device.type == "cuda"
    assert fake_chronos.last_model_name == "amazon/chronos-2"
    assert fake_chronos.last_kwargs == {"device_map": "cuda", "dtype": torch.float32}


def test_chronos_extractor_cpu_stages_embed_inputs_for_non_cpu_pipeline_device(
    monkeypatch,
) -> None:
    strict_pipeline = _StrictCpuInputChronosPipeline()
    monkeypatch.setattr(
        chronos_module,
        "_load_pipeline",
        lambda *args, **kwargs: strict_pipeline,
    )

    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 5),
        dtype=np.float32,
    )
    extractor = ChronosExtractor(
        observation_space,
        feature_names=["Open", "High", "Low", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )

    assert [call.device.type for call in strict_pipeline.calls] == ["cpu"]

    batch = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
    features = extractor(batch)

    assert features.shape == (2, 4)
    assert [call.device.type for call in strict_pipeline.calls] == ["cpu", "cpu"]
    assert [tuple(call.shape) for call in strict_pipeline.calls] == [
        (1, 4, 2),
        (2, 4, 2),
    ]


def test_chronos_embedder_pooling_last_returns_last_token(fake_chronos) -> None:
    embedder = ChronosEmbedder(pooling="last")

    embeddings = embedder.embed_windows(
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        as_tensor=True,
    )

    expected = torch.tensor([[10.5, 9.5, 11.5, 15.0]], dtype=torch.float32)
    torch.testing.assert_close(embeddings, expected)


def test_chronos_embedder_accepts_single_2d_window(fake_chronos) -> None:
    embedder = ChronosEmbedder()

    embeddings = embedder.embed_windows(
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        as_tensor=True,
    )

    assert embeddings.shape == (1, 4)
    assert fake_chronos.last_pipeline is not None
    assert fake_chronos.last_pipeline.calls[-1].shape == (1, 4, 3)


def test_chronos_extractor_accepts_flat_backward_compatible_inputs(
    fake_chronos,
) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(12,),
        dtype=np.float32,
    )
    extractor = ChronosExtractor(
        observation_space,
        lookback=4,
        selected_indices=[0, 2],
    )

    batch = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    features = extractor(batch)

    assert features.shape == (2, 4)
    assert extractor.lookback == 4
    assert extractor.n_features == 3
    assert fake_chronos.last_pipeline is not None
    assert fake_chronos.last_pipeline.calls[-1].shape == (2, 4, 2)


def test_chronos_extractor_requires_lookback_for_flat_inputs(fake_chronos) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(12,),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="requires lookback"):
        ChronosExtractor(observation_space)


def test_chronos_extractor_rejects_invalid_selection_configuration(
    fake_chronos,
) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 5),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="either selected_columns or selected_indices"):
        ChronosExtractor(
            observation_space,
            feature_names=["Open", "High", "Low", "Close", "Volume"],
            selected_columns=["Close"],
            selected_indices=[3],
        )

    with pytest.raises(ValueError, match="selected_columns requires feature_names"):
        ChronosExtractor(
            observation_space,
            selected_columns=["Close"],
        )


def test_chronos_embedder_transform_dataframe_requires_pandas(
    fake_chronos,
    monkeypatch,
) -> None:
    embedder = ChronosEmbedder()

    def _raise_import_error():
        raise ImportError("pandas is not installed")

    original_import = __import__

    def _patched_import(name, *args, **kwargs):
        if name == "pandas":
            _raise_import_error()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _patched_import)

    with pytest.raises(ImportError, match="pandas"):
        embedder.transform_dataframe(object(), lookback=3)


def test_embed_dataframe_is_exported_from_extractors(fake_chronos) -> None:
    assert exported_embed_dataframe is embed_dataframe


def test_embed_dataframe_returns_trimmed_dataframe(
    fake_chronos,
) -> None:
    df = _make_chronos_dataframe()
    offline_df = embed_dataframe(
        df,
        lookback=3,
        frame_bound=(3, len(df)),
        feature_columns=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )

    embedding_columns = [
        "chronos_0",
        "chronos_1",
        "chronos_2",
        "chronos_3",
    ]
    assert list(offline_df.filter(like="chronos_").columns) == embedding_columns
    assert len(offline_df) == 3
    np.testing.assert_allclose(
        offline_df.filter(like="chronos_").to_numpy(dtype=np.float32),
        offline_df.loc[:, embedding_columns].to_numpy(dtype=np.float32),
    )
    assert fake_chronos.last_pipeline is not None
    assert fake_chronos.last_pipeline.calls[-1].shape == (3, 3, 2)


def test_embed_dataframe_can_drop_embedded_feature_columns(fake_chronos) -> None:
    df = _make_chronos_dataframe()

    offline_df = embed_dataframe(
        df,
        lookback=3,
        frame_bound=(3, len(df)),
        feature_columns=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
        drop_feature_columns=True,
    )

    assert list(offline_df.columns) == ["chronos_0", "chronos_1", "chronos_2", "chronos_3"]


@pytest.mark.parametrize(
    ("lookback", "frame_bound", "match"),
    [
        (0, (0, 5), "lookback must be greater than 0"),
        (3, (2, 5), r"frame_bound\[0\] must be at least lookback=3"),
        (3, (3, 3), r"frame_bound must satisfy frame_bound\[0\] < frame_bound\[1\]"),
        (3, (3, 6), r"frame_bound\[1\] must be <= len\(df\)=5"),
    ],
)
def test_embed_dataframe_validates_frame_bounds(
    fake_chronos,
    lookback: int,
    frame_bound: Sequence[int],
    match: str,
) -> None:
    df = _make_chronos_dataframe()

    with pytest.raises(ValueError, match=match):
        embed_dataframe(
            df,
            lookback=lookback,
            frame_bound=frame_bound,
            feature_columns=["Open", "Close", "Volume"],
        )


def test_chronos_embedder_dataframe_alignment_matches_extractor(
    fake_chronos,
) -> None:
    df = _make_chronos_dataframe()
    lookback = 3

    embedder = ChronosEmbedder(
        feature_names=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )
    transformed = embedder.transform_dataframe(
        df,
        lookback=lookback,
        columns=["Open", "Close", "Volume"],
    )

    embedding_columns = [column for column in transformed.columns if column.startswith("chronos_")]
    assert len(embedding_columns) == 4
    assert transformed.loc[: lookback - 2, embedding_columns].isna().all().all()

    windows = np.stack(
        [
            df.iloc[idx : idx + lookback][["Open", "Close", "Volume"]].to_numpy(
                dtype=np.float32
            )
            for idx in range(len(df) - lookback + 1)
        ],
        axis=0,
    )
    extractor = ChronosExtractor(
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, 3), dtype=np.float32),
        feature_names=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )

    online_embeddings = extractor(torch.from_numpy(windows)).detach().numpy()
    offline_embeddings = transformed.loc[lookback - 1 :, embedding_columns].to_numpy(
        dtype=np.float32
    )

    np.testing.assert_allclose(offline_embeddings, online_embeddings)


def test_chronos_embedder_transform_dataframe_progress_bar_matches_default(
    fake_chronos,
    monkeypatch,
) -> None:
    df = _make_chronos_dataframe()
    _install_fake_tqdm(monkeypatch)

    base_embedder = ChronosEmbedder(
        feature_names=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )
    progress_embedder = ChronosEmbedder(
        feature_names=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )

    transformed = base_embedder.transform_dataframe(
        df,
        lookback=3,
        columns=["Open", "Close", "Volume"],
    )
    transformed_with_progress = progress_embedder.transform_dataframe(
        df,
        lookback=3,
        columns=["Open", "Close", "Volume"],
        progress_bar=True,
    )

    np.testing.assert_allclose(
        transformed_with_progress.filter(like="chronos_").to_numpy(dtype=np.float32),
        transformed.filter(like="chronos_").to_numpy(dtype=np.float32),
        equal_nan=True,
    )


def test_chronos_embedder_transform_dataframe_progress_bar_batches_windows(
    fake_chronos,
    monkeypatch,
) -> None:
    df = _make_chronos_dataframe()
    bars = _install_fake_tqdm(monkeypatch)
    monkeypatch.setattr(chronos_module, "_DATAFRAME_PROGRESS_BATCH_SIZE", 2)

    embedder = ChronosEmbedder(
        feature_names=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )
    transformed = embedder.transform_dataframe(
        df,
        lookback=3,
        columns=["Open", "Close", "Volume"],
        progress_bar=True,
    )

    embedding_columns = [column for column in transformed.columns if column.startswith("chronos_")]
    assert len(embedding_columns) == 4
    assert fake_chronos.last_pipeline is not None
    assert [tuple(call.shape) for call in fake_chronos.last_pipeline.calls] == [
        (2, 3, 2),
        (1, 3, 2),
    ]
    assert len(bars) == 1
    assert bars[0].total == 3
    assert bars[0].unit == "window"
    assert bars[0].desc == "Chronos embeddings"
    assert bars[0].dynamic_ncols is True
    assert bars[0].updates == [2, 1]
    assert bars[0].closed is True


def test_chronos_embedder_transform_dataframe_progress_bar_requires_tqdm(
    fake_chronos,
    monkeypatch,
) -> None:
    df = _make_chronos_dataframe()
    embedder = ChronosEmbedder(
        feature_names=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
    )

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
        embedder.transform_dataframe(
            df,
            lookback=3,
            columns=["Open", "Close", "Volume"],
            progress_bar=True,
        )

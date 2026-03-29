from __future__ import annotations

import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from crosslearn.extractors.base import BaseFeaturesExtractor
from crosslearn.extractors.chronos import ChronosEmbedder, ChronosExtractor


class _FakeChronosPipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls: list[torch.Tensor] = []

    def embed(self, context):
        context_t = torch.as_tensor(context, dtype=torch.float32)
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


class _FakeBaseChronosPipeline:
    last_model_name: str | None = None
    last_kwargs: dict | None = None
    last_pipeline: _FakeChronosPipeline | None = None

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        cls.last_model_name = model_name
        cls.last_kwargs = kwargs
        cls.last_pipeline = _FakeChronosPipeline()
        return cls.last_pipeline


@pytest.fixture
def fake_chronos(monkeypatch):
    module = types.ModuleType("chronos")
    module.BaseChronosPipeline = _FakeBaseChronosPipeline
    monkeypatch.setitem(sys.modules, "chronos", module)
    _FakeBaseChronosPipeline.last_model_name = None
    _FakeBaseChronosPipeline.last_kwargs = None
    _FakeBaseChronosPipeline.last_pipeline = None
    return _FakeBaseChronosPipeline


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
) -> None:
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


def test_chronos_embedder_dataframe_alignment_matches_extractor(
    fake_chronos,
) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Close": [2.0, 3.0, 4.0, 5.0, 6.0],
            "Volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
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

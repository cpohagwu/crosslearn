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
from crosslearn.envs import WalkForwardChronosWrapper


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


class _SingleObservationSequenceEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self._observations = np.array(
            [
                [1.0, 10.0],
                [2.0, 11.0],
                [3.0, 12.0],
                [4.0, 13.0],
            ],
            dtype=np.float32,
        )
        self._index = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._index = 0
        return self._observations[self._index].copy(), {}

    def step(self, action: int):
        self._index = min(self._index + 1, len(self._observations) - 1)
        terminated = self._index >= len(self._observations) - 1
        return (
            self._observations[self._index].copy(),
            float(action),
            terminated,
            False,
            {"index": self._index},
        )


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
            summary = torch.stack(
                [
                    context_t.mean(dim=1),
                    context_t[:, 0],
                    context_t[:, -1],
                    context_t[:, -1],
                ],
                dim=-1,
            )
        else:
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


class _FakeChronos2FastModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.chronos_config = types.SimpleNamespace(
            output_patch_size=1,
            max_output_patches=1,
        )
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def encode(self, *, context: torch.Tensor, group_ids: torch.Tensor):
        self.calls.append((context.detach().clone(), group_ids.detach().clone()))
        summary = torch.stack(
            [context.mean(dim=-1), group_ids.to(dtype=torch.float32)],
            dim=-1,
        )
        hidden_states = torch.stack([summary, summary + 1.0], dim=1)
        return (hidden_states,), (torch.zeros(context.shape[0]), torch.ones(context.shape[0]))


class _FakeChronos2FastPipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = _FakeChronos2FastModel()

    def embed(self, context, *args, **kwargs):
        raise AssertionError("Chronos-2 direct path should bypass pipeline.embed().")


class _FakeChronosBoltPipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = _FakeChronosBoltModel()

    def embed(self, context, batch_size: int = 256):
        raise AssertionError("Chronos-Bolt direct path should bypass pipeline.embed().")


class _FakeChronosBoltModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def encode(self, *, context: torch.Tensor, mask: torch.Tensor):
        self.calls.append((context.detach().clone(), mask.detach().clone()))
        summary = torch.stack(
            [context.mean(dim=-1), context[:, -1]],
            dim=-1,
        )
        return torch.stack([summary, summary + 1.0], dim=1), None, None, None


class _FakeChronosT5Tokenizer:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def context_input_transform(self, context: torch.Tensor):
        self.calls.append(context.detach().clone())
        token_ids = context.to(dtype=torch.long)
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        return token_ids, attention_mask, torch.ones(context.shape[0])


class _FakeChronosT5Model:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def encode(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        summary = torch.stack(
            [
                input_ids.to(dtype=torch.float32).mean(dim=-1),
                input_ids[:, -1].to(dtype=torch.float32),
            ],
            dim=-1,
        )
        return torch.stack([summary, summary + 1.0], dim=1)


class _FakeChronosT5Pipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = _FakeChronosT5Tokenizer()
        self.model = _FakeChronosT5Model()

    def embed(self, context, *args, **kwargs):
        raise AssertionError("Chronos T5 direct path should bypass pipeline.embed().")


class _CountingChronosPipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls: list[torch.Tensor] = []

    def embed(self, context):
        context_t = torch.as_tensor(context, dtype=torch.float32)
        self.calls.append(context_t.clone())
        if context_t.ndim == 2:
            context_t = context_t.unsqueeze(1)
        summary = torch.stack(
            [
                context_t.mean(dim=(1, 2)),
                context_t[:, 0, :].mean(dim=1),
                context_t[:, -1, :].mean(dim=1),
            ],
            dim=-1,
        )
        return torch.stack([summary, summary + 1.0], dim=1), {"counting": True}


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
    assert fake_chronos.last_pipeline.calls[-1].shape == (2, 2, 4)


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
        (1, 2, 4),
        (2, 2, 4),
    ]


def test_chronos2_model_names_use_direct_encode_with_grouped_variates(
    monkeypatch,
) -> None:
    pipeline = _FakeChronos2FastPipeline()
    monkeypatch.setattr(
        chronos_module,
        "_load_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    embedder = ChronosEmbedder(
        model_name="amazon/chronos-2",
        selected_indices=[1, 2],
    )

    windows = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    embeddings = embedder.embed_windows(
        windows,
        lookback=4,
        n_features=3,
        as_tensor=True,
    )

    assert embeddings.shape == (2, 2)
    assert len(pipeline.model.calls) == 1
    context, group_ids = pipeline.model.calls[0]
    assert context.shape == (4, 4)
    torch.testing.assert_close(
        context,
        torch.tensor(
            [
                [1.0, 4.0, 7.0, 10.0],
                [2.0, 5.0, 8.0, 11.0],
                [13.0, 16.0, 19.0, 22.0],
                [14.0, 17.0, 20.0, 23.0],
            ]
        ),
    )
    torch.testing.assert_close(group_ids, torch.tensor([0, 0, 1, 1]))


def test_chronos_bolt_model_names_use_direct_encode(
    monkeypatch,
) -> None:
    pipeline = _FakeChronosBoltPipeline()
    monkeypatch.setattr(
        chronos_module,
        "_load_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    embedder = ChronosEmbedder(
        model_name="amazon/chronos-bolt-base",
        selected_indices=[0, 2],
    )

    windows = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    embeddings = embedder.embed_windows(
        windows,
        lookback=4,
        n_features=3,
        as_tensor=True,
    )

    assert embeddings.shape == (2, 2)
    assert len(pipeline.model.calls) == 1
    context, mask = pipeline.model.calls[0]
    assert context.shape == (4, 4)
    assert mask.shape == context.shape
    assert mask.all()
    torch.testing.assert_close(
        context,
        torch.tensor(
            [
                [0.0, 3.0, 6.0, 9.0],
                [2.0, 5.0, 8.0, 11.0],
                [12.0, 15.0, 18.0, 21.0],
                [14.0, 17.0, 20.0, 23.0],
            ]
        ),
    )


def test_chronos_t5_model_names_use_direct_encode(monkeypatch) -> None:
    pipeline = _FakeChronosT5Pipeline()
    monkeypatch.setattr(
        chronos_module,
        "_load_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    embedder = ChronosEmbedder(
        model_name="amazon/chronos-t5-small",
        selected_indices=[0, 2],
    )

    windows = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    embeddings = embedder.embed_windows(
        windows,
        lookback=4,
        n_features=3,
        as_tensor=True,
    )

    assert embeddings.shape == (2, 2)
    assert len(pipeline.tokenizer.calls) == 1
    assert len(pipeline.model.calls) == 1
    token_context = pipeline.tokenizer.calls[0]
    input_ids, attention_mask = pipeline.model.calls[0]
    assert token_context.shape == (4, 4)
    torch.testing.assert_close(input_ids, token_context.to(dtype=torch.long))
    assert attention_mask.shape == input_ids.shape


def test_amazon_chronos_unknown_direct_adapter_warns_and_falls_back(
    monkeypatch,
) -> None:
    pipeline = _CountingChronosPipeline()
    monkeypatch.setattr(
        chronos_module,
        "_load_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    embedder = ChronosEmbedder(model_name="amazon/chronos-future")

    with pytest.warns(RuntimeWarning, match="falling back"):
        embeddings = embedder.embed_windows(
            torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
            lookback=4,
            n_features=3,
            as_tensor=True,
        )

    assert embeddings.shape == (1, 3)
    assert embedder.last_embedding_path == "pipeline_embed"
    assert "no direct adapter" in str(embedder.last_fallback_reason)


def test_chronos_extractor_excludes_offline_embedding_knobs(fake_chronos) -> None:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 3),
        dtype=np.float32,
    )

    with pytest.raises(TypeError, match="use_fast_path"):
        ChronosExtractor(observation_space, use_fast_path=False)

    with pytest.raises(TypeError, match="embed_batch_size"):
        ChronosExtractor(observation_space, embed_batch_size=16)


def test_chronos_embedder_cache_reuses_repeated_windows(monkeypatch) -> None:
    pipeline = _CountingChronosPipeline()
    monkeypatch.setattr(
        chronos_module,
        "_load_pipeline",
        lambda *args, **kwargs: pipeline,
    )
    embedder = ChronosEmbedder(
        model_name="amazon/chronos-2",
        selected_indices=[0, 1],
        cache_size=4,
    )

    windows = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    first = embedder.embed_windows(
        windows,
        lookback=4,
        n_features=3,
        as_tensor=True,
    )
    second = embedder.embed_windows(
        windows,
        lookback=4,
        n_features=3,
        as_tensor=True,
    )
    mixed = torch.cat([windows[1:2], windows[0:1] + 100.0], dim=0)
    third = embedder.embed_windows(
        mixed,
        lookback=4,
        n_features=3,
        as_tensor=True,
    )

    torch.testing.assert_close(first, second)
    assert third.shape == first.shape
    assert [tuple(call.shape) for call in pipeline.calls] == [
        (2, 2, 4),
        (1, 2, 4),
    ]


def test_walkforward_chronos_wrapper_waits_for_min_history(fake_chronos) -> None:
    wrapped = WalkForwardChronosWrapper(
        _SingleObservationSequenceEnv(),
        lookback=3,
        feature_names=["a", "b"],
        min_history=3,
        model_name="custom/fake",
    )

    obs, _ = wrapped.reset()
    assert obs.shape == wrapped.observation_space.shape
    np.testing.assert_allclose(obs, np.zeros_like(obs))

    obs, *_ = wrapped.step(0)
    np.testing.assert_allclose(obs, np.zeros_like(obs))

    obs, *_ = wrapped.step(1)
    assert not np.allclose(obs, np.zeros_like(obs))
    assert fake_chronos.last_pipeline is not None
    assert [tuple(call.shape) for call in fake_chronos.last_pipeline.calls] == [
        (2, 3),
        (2, 3),
    ]


def test_walkforward_chronos_wrapper_allows_min_history_above_lookback(
    fake_chronos,
) -> None:
    wrapped = WalkForwardChronosWrapper(
        _SingleObservationSequenceEnv(),
        lookback=2,
        feature_names=["a", "b"],
        min_history=3,
        model_name="custom/fake",
    )

    obs, _ = wrapped.reset()
    np.testing.assert_allclose(obs, np.zeros_like(obs))

    obs, *_ = wrapped.step(0)
    np.testing.assert_allclose(obs, np.zeros_like(obs))

    obs, *_ = wrapped.step(1)
    assert not np.allclose(obs, np.zeros_like(obs))
    assert fake_chronos.last_pipeline is not None
    assert [tuple(call.shape) for call in fake_chronos.last_pipeline.calls] == [
        (2, 2),
        (2, 2),
    ]


def test_walkforward_chronos_wrapper_supports_expanding_history(
    fake_chronos,
) -> None:
    wrapped = WalkForwardChronosWrapper(
        _SingleObservationSequenceEnv(),
        lookback=2,
        feature_names=["a", "b"],
        min_history=2,
        expanding_window=True,
        model_name="custom/fake",
    )

    wrapped.reset()
    wrapped.step(0)
    wrapped.step(0)

    assert fake_chronos.last_pipeline is not None
    assert [tuple(call.shape) for call in fake_chronos.last_pipeline.calls] == [
        (2, 2),
        (2, 2),
        (2, 3),
    ]


def test_chronos_embedder_pooling_last_returns_last_token(fake_chronos) -> None:
    embedder = ChronosEmbedder(pooling="last")

    embeddings = embedder.embed_windows(
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        as_tensor=True,
    )

    expected = torch.tensor([[10.5, 6.0, 15.0, 11.5]], dtype=torch.float32)
    torch.testing.assert_close(embeddings, expected)


def test_chronos_embedder_accepts_single_2d_window(fake_chronos) -> None:
    embedder = ChronosEmbedder()

    embeddings = embedder.embed_windows(
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        as_tensor=True,
    )

    assert embeddings.shape == (1, 4)
    assert fake_chronos.last_pipeline is not None
    assert fake_chronos.last_pipeline.calls[-1].shape == (1, 3, 4)


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
    assert fake_chronos.last_pipeline.calls[-1].shape == (2, 2, 4)


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
        feature_names=["Open", "Close", "Volume"],
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
    assert fake_chronos.last_pipeline.calls[-1].shape == (3, 2, 3)


def test_embed_dataframe_uses_all_numeric_features_when_names_are_omitted(
    fake_chronos,
) -> None:
    df = _make_chronos_dataframe()
    df["Symbol"] = ["TEST"] * len(df)

    offline_df = embed_dataframe(
        df,
        lookback=3,
        frame_bound=(3, len(df)),
        selected_columns=["Open", "Volume"],
    )

    assert "Symbol" in offline_df.columns
    assert fake_chronos.last_pipeline is not None
    assert fake_chronos.last_pipeline.calls[-1].shape == (3, 2, 3)


def test_embed_dataframe_can_drop_embedded_feature_names(fake_chronos) -> None:
    df = _make_chronos_dataframe()

    offline_df = embed_dataframe(
        df,
        lookback=3,
        frame_bound=(3, len(df)),
        feature_names=["Open", "Close", "Volume"],
        selected_columns=["Close", "Volume"],
        drop_feature_names=True,
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
            feature_names=["Open", "Close", "Volume"],
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
        (2, 2, 3),
        (1, 2, 3),
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

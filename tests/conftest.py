from __future__ import annotations

import sys
import types

import pytest
import torch


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

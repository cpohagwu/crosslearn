"""Benchmark CrossLearn Chronos direct encoding against pipeline.embed fallback.

This script is intentionally small and local. It uses CrossLearn's public
``ChronosEmbedder.embed_windows(...)`` path for direct encoding, then calls the
private fallback helper for an apples-to-apples latency comparison on the same
normalized windows. Caching is disabled in both paths.

Example:

    python scripts/benchmark_chronos_embedding.py \
        --model-name amazon/chronos-2 \
        --batch-size 8 \
        --lookback 64 \
        --n-features 5 \
        --device-map cuda
"""

from __future__ import annotations

import argparse
import time

import torch

from crosslearn.extractors.chronos import ChronosEmbedder, _normalize_window_batch


def _parse_dtype(value: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"Unsupported dtype {value!r}. Choose from {sorted(dtypes)}."
        ) from exc


def _time_call(fn, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="amazon/chronos-2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.float32)
    args = parser.parse_args()

    windows = torch.randn(
        args.batch_size,
        args.lookback,
        args.n_features,
        dtype=torch.float32,
    )
    embedder = ChronosEmbedder(
        model_name=args.model_name,
        device_map=args.device_map,
        dtype=args.dtype,
        cache_size=0,
        embed_batch_size=args.batch_size * args.n_features,
    )
    normalized_windows, _, _ = _normalize_window_batch(
        windows,
        lookback=args.lookback,
        n_features=args.n_features,
    )

    def direct() -> None:
        embedder._embed_direct_amazon(normalized_windows)

    def fallback() -> None:
        embedder._embed_with_pipeline(normalized_windows)

    direct()
    fallback()
    direct_seconds = _time_call(direct, args.repeats)
    fallback_seconds = _time_call(fallback, args.repeats)

    print(f"model_name={args.model_name}")
    print(f"shape=({args.batch_size}, {args.lookback}, {args.n_features})")
    print(f"direct_avg_seconds={direct_seconds:.6f}")
    print(f"pipeline_embed_avg_seconds={fallback_seconds:.6f}")
    if direct_seconds > 0:
        print(f"fallback_over_direct={fallback_seconds / direct_seconds:.2f}x slower")


if __name__ == "__main__":
    main()

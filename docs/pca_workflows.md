# Walk-Forward PCA Workflows

This document explains how walk-forward PCA is exposed in CrossLearn, how it
changes the data, and how to combine it with Chronos in both offline and
constrained online workflows.

## What PCA Does

PCA does not keep the original feature units. It rotates the data into a new
coordinate system whose axes are the principal directions of variation in the
fit data.

- the original columns stay in their original units before PCA
- the PCA output columns are principal-component scores, not raw measurements
- if `standardize=True`, those scores are computed after walk-forward centering
  and walk-forward division by the fit-window standard deviation
- if `standardize=False`, the data is still centered before PCA, but not
  scaled by standard deviation

In other words, `pca_0`, `pca_1`, and so on are latent coordinates. They
should be interpreted as projections onto the fitted component directions, not
as renamed source columns.

## Why Full-Sample PCA Leaks in Time Series

If PCA is fit on the entire dataset first and then used everywhere, the
projection for early timestamps already contains information from future rows.
That is leakage.

CrossLearn avoids that by using a walk-forward procedure:

1. fit the initial PCA on the first `warmup` rows
2. choose the smallest fixed `n_components` whose cumulative explained variance
   reaches `explained_variance_threshold`
3. transform row `warmup` using only the PCA fit from rows `0 .. warmup - 1`
4. refit the mean, optional standard deviation, and PCA loadings on the chosen
   history window
5. transform the next row
6. repeat until the end

The component count stays fixed after the initial warmup fit, but the scaling
statistics and loadings are recomputed at every step using past rows only.

## Solvers and History Windows

CrossLearn exposes two exact PCA backends:

- `solver="svd"`:
  direct singular-value decomposition of the transformed history matrix
- `solver="covariance_eigh"`:
  eigendecomposition of the covariance or correlation matrix derived from the
  same history matrix

CrossLearn also exposes two history policies through `expanding_warmup`:

- `expanding_warmup=True`:
  fit PCA on all available past rows after warmup
- `expanding_warmup=False`:
  fit PCA on exactly the last `warmup` rows before each next-row projection

These two knobs are independent. For example:

- `solver="svd", expanding_warmup=True` is the default expanding SVD workflow
- `solver="svd", expanding_warmup=False` is rolling-window SVD
- `solver="covariance_eigh", expanding_warmup=True` is expanding covariance PCA
- `solver="covariance_eigh", expanding_warmup=False` is rolling covariance PCA

## Why `covariance_eigh` Uses a Square Matrix Without Padding

The covariance solver does not pad the history matrix.

If the centered or standardized history matrix is:

```text
X shape = (n_samples, n_features)
```

then the covariance-like matrix used by the eigendecomposition path is:

```text
X^T X / (n_samples - 1)
```

and that matrix always has shape:

```text
(n_features, n_features)
```

So it is square by construction. No zero-padding is needed.

## Precision Tradeoffs: `svd` vs `covariance_eigh`

In exact arithmetic, `svd` and `covariance_eigh` recover the same PCA subspace.
In floating-point arithmetic, they are close but not bit-identical.

Reference behavior:

- `svd` is the more numerically stable path
- `covariance_eigh` is usually closer to the same answer on well-conditioned
  problems, but it is more sensitive to conditioning
- component signs may differ even when the underlying subspace is the same
- near-tied components may rotate within the same subspace

This difference is more noticeable when:

- the data is poorly conditioned
- trailing components are very small
- `compute_dtype=torch.float32` is used

See [Troubleshooting](#troubleshooting) for the practical implications of
`covariance_eigh` with `compute_dtype=torch.float32` on CUDA.

`compute_dtype=torch.float64` remains the default because it is the closest
match to the current stable behavior.

## Performance Considerations

Chronos embedding and walk-forward PCA behave differently from a hardware point
of view:

- `embed_dataframe(...)` can batch many windows together, so GPU utilization is
  naturally high when Chronos runs on CUDA
- walk-forward PCA is still chronological by design, because each step must
  refit on past rows only

CrossLearn still uses CUDA when available for the heavy PCA math inside each
offline batch or online step:

- `WalkForwardPCATransformer(..., device="auto")`
- `walkforward_pca_dataframe(..., device="auto")`
- `WalkForwardChronosPCAWrapper(..., device_map="auto")`

That means the decomposition, centering, scaling, and projection work can run
on GPU even though the chronology itself remains sequential.

### When CPU Can Be Faster

It is normal for this PCA path to run slower on GPU than on CPU for some
workloads.

Common reasons:

- the heavy operation is repeated exact decomposition, not one large GEMM
- `compute_dtype=torch.float64` is the default
- many consumer GPUs have weak `float64` throughput
- smaller history windows or smaller feature counts may not amortize GPU
  overhead well
- `batch_size` may need to stay small for memory reasons

### When GPU Is More Likely to Help

GPU utilization is more likely to improve when:

- `solver="covariance_eigh"` is acceptable for the workload
- `expanding_warmup=False` gives a fixed history height of `warmup`
- `compute_dtype=torch.float32` is acceptable for the workload
- `batch_size` is large enough to keep the device busy without exceeding
  available memory
- feature width is large enough for the decomposition to dominate transfer and
  launch overhead

These are tendencies, not guarantees. The best-performing configuration is
workload- and hardware-dependent.

For difficult batches, maximizing GPU utilization with
`solver="covariance_eigh"` and `compute_dtype=torch.float32` may also increase
the chance of eigendecomposition convergence failures. See
[Troubleshooting](#troubleshooting).

## Memory and `batch_size`

`batch_size` controls how many chronological PCA windows are processed together
in the offline helper path.

For the SVD backend with expanding history, the dominant tensor is typically
shaped like:

```text
(batch_size, max_history_in_chunk, n_features)
```

For the covariance backend, the dominant tensor is typically shaped like:

```text
(batch_size, n_features, n_features)
```

Both can be large, especially in `float64`. The easiest safety knob is usually
`batch_size`.

Implications:

- smaller `batch_size` lowers RAM or VRAM pressure
- larger `batch_size` can improve throughput if memory allows it
- late chunks in expanding mode are often the heaviest, because the history
  window has grown the most

If a CPU or GPU memory allocation fails, lowering `batch_size` is the first
thing to try.

## Maximizing GPU Utilization

The following settings are usually the most relevant GPU-utilization levers:

- `solver="covariance_eigh"`:
  the decomposition is applied to a square `(n_features, n_features)` matrix
- `expanding_warmup=False`:
  every refit uses the same history length `warmup`
- `compute_dtype=torch.float32`:
  often much faster on consumer GPUs than `float64`
- larger `batch_size`:
  useful when memory allows it

Practical reference points:

- if Chronos benefits from GPU but PCA does not, use GPU for Chronos and CPU
  for PCA
- the offline workflow already supports that split through
  `embed_dataframe(..., device_map="auto")` followed by
  `walkforward_pca_dataframe(..., device="cpu")`
- the online wrapper supports the same idea through `device_map` for Chronos
  and `pca_device` for PCA

`solver="covariance_eigh" + compute_dtype=torch.float32 + CUDA` is a
performance-oriented but discouraged configuration because it is more likely to
encounter convergence failures on ill-conditioned batches. See
[Troubleshooting](#troubleshooting).

## Core APIs

### `WalkForwardPCATransformer`

Use this class when you want direct array-level control.

```python
import torch

from crosslearn.extractors import WalkForwardPCATransformer

transformer = WalkForwardPCATransformer(
    warmup=500,
    explained_variance_threshold=0.99,
    standardize=True,
    solver="svd",
    expanding_warmup=True,
    compute_dtype=torch.float64,
    device="auto",
    batch_size=256,
)

projected = transformer.walkforward_transform(values)
print(transformer.n_components_)
```

Behavior:

- `warmup` must be at least `2`
- `n_components_` is chosen once from the initial warmup fit
- `standardize=True` recomputes mean and standard deviation walk-forward
- `standardize=False` still recomputes the mean walk-forward, but skips
  division by standard deviation
- `solver="svd"` is the default stable path
- `solver="covariance_eigh"` uses a square covariance or correlation matrix
- `expanding_warmup=False` means rolling PCA on exactly the last `warmup` rows
- `compute_dtype=torch.float32` is the faster but less stable option
- `batch_size` controls how many chronological PCA windows are processed per
  offline chunk

### `walkforward_pca_dataframe`

Use this helper when your data already lives in a dataframe.

```python
import torch

from crosslearn.extractors import walkforward_pca_dataframe

pca_df = walkforward_pca_dataframe(
    df,
    feature_columns=["feature_a", "feature_b", "feature_c"],
    warmup=500,
    explained_variance_threshold=0.99,
    standardize=True,
    solver="covariance_eigh",
    expanding_warmup=False,
    compute_dtype=torch.float32,
    device="auto",
    batch_size=128,
    output_prefix="pca_",
    drop_feature_columns=False,
    return_transformed_warmup=True,
    trim_warmup=False,
    progress_bar=True,
)
```

Key arguments:

- `feature_columns`: source columns to reduce
- `warmup`: number of rows required before the first future-safe next-row
  projection is available
- `solver`: `svd` or `covariance_eigh`
- `expanding_warmup`: expanding history vs rolling history
- `compute_dtype`: internal PCA precision
- `device`: torch device for PCA math
- `batch_size`: number of chronological PCA windows processed together in the
  offline path
- `return_transformed_warmup`: when `True`, backfills the first `warmup` PCA
  rows with the initial warmup fit-transform
- `trim_warmup`: drops the first `warmup` rows entirely; when `False`, the
  helper keeps the full dataframe length

## Offline Chronos + PCA

Chronos and PCA are intentionally separate steps:

1. create aligned Chronos embeddings with `embed_dataframe(...)`
2. run walk-forward PCA on the resulting `chronos_*` columns

```python
import torch

from crosslearn.extractors import embed_dataframe, walkforward_pca_dataframe

embedded_df = embed_dataframe(
    df,
    lookback=30,
    frame_bound=(30, len(df)),
    feature_columns=["Open", "High", "Low", "Close", "Volume"],
    selected_columns=["Close", "Volume"],
    progress_bar=True,
)

pca_df = walkforward_pca_dataframe(
    embedded_df,
    feature_columns=embedded_df.filter(like="chronos_").columns.tolist(),
    warmup=500,
    explained_variance_threshold=0.99,
    standardize=True,
    solver="svd",
    expanding_warmup=True,
    compute_dtype=torch.float64,
    device="cpu",
    batch_size=64,
    output_prefix="pca_",
    drop_feature_columns=True,
    trim_warmup=True,
    progress_bar=True,
)
```

This gives you a new dataframe whose PCA columns are leakage-safe by
construction from row `warmup` onward. If `trim_warmup=True`, the dataset
becomes shorter by `warmup` rows because only future-safe next-row projections
are kept. If `trim_warmup=False`, the helper keeps the original dataframe
length and, by default, fills the first `warmup` PCA rows with retrospective
scores from the initial warmup fit. Set `return_transformed_warmup=False` to
leave those warmup rows as `NaN` instead.

## Constrained Online Chronos + PCA

Adaptive PCA is not built into `ChronosExtractor`.

That is intentional. In CrossLearn, the extractor is used both during rollout
collection and again during policy updates on stored observations. A mutable
extractor would allow the same observation to map to different features
depending on call order.

Use `WalkForwardChronosPCAWrapper` instead when the environment is sequential
and observation history is determined by time index:

```python
import torch

from crosslearn.envs import WalkForwardChronosPCAWrapper

wrapped_env = WalkForwardChronosPCAWrapper(
    env,
    lookback=30,
    warmup=500,
    feature_columns=["Open", "High", "Low", "Close", "Volume"],
    selected_columns=["Close", "Volume"],
    solver="covariance_eigh",
    expanding_warmup=False,
    compute_dtype=torch.float32,
    device_map="auto",
    pca_device="cpu",
)
```

Important constraint:

- by design, the first `lookback + warmup` observations are skipped
- the first agent-visible observation is the next one after that skipped prefix
- `frame_bound[0]` must therefore be at least `lookback + warmup`
- `frame_bound[1]` must be greater than `frame_bound[0]`
- for a plain dataset with no extra slicing, that means you need at least
  `lookback + warmup + 1` rows to get one projected observation

Runtime behavior:

- at wrapper construction, CrossLearn embeds only the warmup windows needed to
  determine the fixed PCA width
- at `reset()`, it embeds the current raw window and projects it with the PCA
  fit from the warmup embedding history
- at each `step()`, it appends the previously used embedding to the PCA history,
  refits PCA on either expanding or rolling history, embeds only the newly
  returned raw window, and projects only that new embedding

This online path deliberately does not prepare future PCA windows in batches.
That batching optimization is reserved for the offline PCA helpers, where the
full chronological matrix is already available.

## Choosing `standardize=True`

`standardize=True` is the default because PCA is sensitive to feature scale.

With Chronos embeddings this matters less than with raw OHLCV or mixed
engineered features, but the same rule applies: if some columns vary much more
than others, PCA without standardization will preferentially align with those
larger-scale dimensions.

When `standardize=True`, CrossLearn recomputes all three pieces walk-forward:

- mean
- standard deviation
- PCA loadings

No future row is used when transforming the current row.

## Troubleshooting

### `_LinAlgError` with `solver="covariance_eigh"` on CUDA

If you see an error like:

```text
torch.linalg.eigh(...): The algorithm failed to converge because the input
matrix is ill-conditioned or has too many repeated eigenvalues
```

that is a known limitation of the covariance solver on some workloads,
especially with:

- `solver="covariance_eigh"`
- `compute_dtype=torch.float32`
- CUDA enabled

Root cause:

- the covariance solver uses batched `torch.linalg.eigh` on covariance or
  correlation matrices
- in `float32`, some of those matrices can become numerically ill-conditioned
  enough for `eigh` to fail to converge
- repeated or near-repeated eigenvalues make this more likely
- `batch_size` changes which chronological windows are grouped together, so
  some values such as `32` or `64` may fail while others do not on the same
  dataset

This is a convergence issue, not a leakage bug and not a chronology bug.

This configuration is discouraged:

- `solver="covariance_eigh" + compute_dtype=torch.float32 + CUDA`

Recommended mitigations, in priority order:

1. switch to `compute_dtype=torch.float64`
2. switch to `solver="svd"`
3. keep Chronos on GPU but run PCA on CPU
4. if the issue appears only for some chunkings, try a different `batch_size`

Split-device examples:

- offline:
  `walkforward_pca_dataframe(..., device="cpu")`
- online:
  `WalkForwardChronosPCAWrapper(..., pca_device="cpu")`

Changing `batch_size` may help because it changes chunk composition, but it is
not a guaranteed fix for convergence failures.

### CPU or GPU Memory Allocation Errors

If you hit a memory allocation error rather than a convergence error, the first
thing to reduce is `batch_size`.

Why this helps:

- smaller `batch_size` lowers peak RAM or VRAM usage
- expanding history chunks are usually heaviest near the end of the dataset
- `float64` uses more memory than `float32`

This is a different problem from `eigh` convergence:

- lowering `batch_size` often helps memory errors
- lowering `batch_size` does not guarantee fixing `covariance_eigh`
  convergence failures

## Related APIs

- [Chronos guide](./chronos.md)
- `ChronosExtractor` for online embedding without adaptive PCA
- `ChronosEmbedder.transform_dataframe(...)` for lower-level aligned embedding
  augmentation

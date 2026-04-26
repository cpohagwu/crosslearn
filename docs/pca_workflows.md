# Walk-Forward PCA Workflows

This document explains how walk-forward PCA is exposed in CrossLearn, how it changes the data, and how to combine it with Chronos in both offline and constrained online workflows.

## What PCA Does

PCA does not keep the original feature units. It rotates the data into a new coordinate system whose axes are the principal directions of variation in the fit data.

- the original columns stay in their original units before PCA
- the PCA output columns are principal-component scores, not raw measurements
- if `standardize=True`, those scores are computed after walk-forward centering and walk-forward division by the fit-window standard deviation
- if `standardize=False`, the data is still centered before PCA, but not scaled by standard deviation

In other words, `pca_0`, `pca_1`, and so on are latent coordinates. They should be interpreted as projections onto the fitted component directions, not as renamed source columns.

## Why Full-Sample PCA Leaks in Time Series

If PCA is fit on the entire dataset first and then used everywhere, the projection for early timestamps already contains information from future rows. That is leakage.

CrossLearn avoids that by using an expanding walk-forward procedure:

1. fit the initial PCA on the first `warmup` rows
2. choose the smallest fixed `n_components` whose cumulative explained variance reaches `explained_variance_threshold`
3. transform row `warmup` using only the PCA fit from rows `0 .. warmup - 1`
4. refit the mean, optional standard deviation, and PCA loadings on rows `0 .. t - 1`
5. transform row `t`
6. repeat until the end

The component count stays fixed after the initial warmup fit, but the scaling statistics and loadings are recomputed walk-forward at every step.

## GPU Acceleration

Chronos embedding and walk-forward PCA behave differently from a hardware point
of view:

- `embed_dataframe(...)` can batch many windows together, so GPU utilization is
  naturally high when Chronos runs on CUDA
- walk-forward PCA is still sequential by design, because each step must refit
  on past rows only

CrossLearn still uses CUDA when available for the heavy PCA math inside each
offline batch or online step:

- `WalkForwardPCATransformer(..., device="auto")`
- `walkforward_pca_dataframe(..., device="auto")`
- `WalkForwardChronosPCAWrapper(..., device_map="auto")`

That means the SVD, centering, scaling, and projection work can run on GPU even
though the chronology itself remains sequential.

For the offline helpers, CrossLearn prepares the walk-forward prefix windows
ahead of time and processes them in batches:

- `WalkForwardPCATransformer(..., batch_size=256)`
- `walkforward_pca_dataframe(..., batch_size=256)`

The online wrapper does not do this. It remains one new embedding and one new
projection at a time by design.

## Core APIs

### `WalkForwardPCATransformer`

Use this class when you want direct array-level control.

```python
from crosslearn.extractors import WalkForwardPCATransformer

transformer = WalkForwardPCATransformer(
    warmup=500,
    explained_variance_threshold=0.99,
    standardize=True,
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
- `standardize=False` still recomputes the mean walk-forward, but skips division by standard deviation
- `device="auto"` prefers CUDA when available
- `batch_size` controls how many prepared walk-forward prefix windows are
  processed per offline batched SVD call
- component signs are aligned against the previous fit to reduce arbitrary sign flips

### `walkforward_pca_dataframe`

Use this helper when your data already lives in a dataframe.

```python
from crosslearn.extractors import walkforward_pca_dataframe

pca_df = walkforward_pca_dataframe(
    df,
    feature_columns=["feature_a", "feature_b", "feature_c"],
    warmup=500,
    explained_variance_threshold=0.99,
    standardize=True,
    device="auto",
    batch_size=256,
    output_prefix="pca_",
    drop_feature_columns=False,
    trim_warmup=False,
    progress_bar=True,
)
```

Key arguments:

- `feature_columns`: source columns to reduce
- `warmup`: number of rows required before the first PCA-transformed row is available
- `standardize`: when `True`, mean and standard deviation are both fit walk-forward
- `device`: torch device for PCA math; `"auto"` prefers CUDA when available
- `batch_size`: number of chronological PCA windows processed together in the
  offline batched path
- `drop_feature_columns`: drops only the columns named in `feature_columns`
- `trim_warmup`: drops the first `warmup` rows instead of leaving `NaN` in the PCA columns
- `progress_bar`: shows a `tqdm` progress bar for the walk-forward PCA pass;
  the total is still measured in rows, but updates happen chunk by chunk

## Offline Chronos + PCA

Chronos and PCA are intentionally separate steps:

1. create aligned Chronos embeddings with `embed_dataframe(...)`
2. run walk-forward PCA on the resulting `chronos_*` columns

```python
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
    device="auto",
    batch_size=256,
    output_prefix="pca_",
    drop_feature_columns=True,
    trim_warmup=True,
    progress_bar=True,
)
```

This gives you a new dataframe whose PCA columns are leakage-safe by construction. The dataset becomes shorter by `warmup` rows because the first PCA-transformed row is only available after the initial fit window.

## Constrained Online Chronos + PCA

Adaptive PCA is not built into `ChronosExtractor`.

That is intentional. In CrossLearn, the extractor is used both during rollout collection and again during policy updates on stored observations. A mutable extractor would allow the same observation to map to different features depending on call order.

Use `WalkForwardChronosPCAWrapper` instead when the environment is sequential and observation history is determined by time index:

```python
from crosslearn.envs import WalkForwardChronosPCAWrapper

wrapped_env = WalkForwardChronosPCAWrapper(
    env,
    lookback=30,
    warmup=500,
    feature_columns=["Open", "High", "Low", "Close", "Volume"],
    selected_columns=["Close", "Volume"],
    device_map="auto",
)
```

Important constraint:

- by design, the first `lookback + warmup` observations are skipped
- the first agent-visible observation is the next one after that skipped prefix
- `frame_bound[0]` must therefore be at least `lookback + warmup`
- `frame_bound[1]` must be greater than `frame_bound[0]`
- for a plain dataset with no extra slicing, that means you need at least `lookback + warmup + 1` rows to get one projected observation

Runtime behavior:

- at wrapper construction, CrossLearn embeds only the warmup windows needed to determine the fixed PCA width
- at `reset()`, it embeds the current raw window and projects it with the PCA fit from the warmup embedding history
- at each `step()`, it appends the previously used embedding to the PCA history, refits PCA on that expanding history, embeds only the newly returned raw window, and projects only that new embedding

This online path deliberately does not prepare future PCA windows in batches.
That batching optimization is reserved for the offline PCA helpers, where the
full chronological matrix is already available.

`device_map` controls both Chronos embedding placement and the internal PCA
math in the wrapper. With `device_map="auto"`, CrossLearn prefers CUDA when it
is available.

That means the first observation returned to the agent is the first post-skip observation, already PCA-ready, and every later observation is computed online from one newly embedded raw window plus an expanding PCA history that only contains past embeddings.

## Choosing `standardize=True`

`standardize=True` is the default because PCA is sensitive to feature scale.

With Chronos embeddings this matters less than with raw OHLCV or mixed engineered features, but the same rule applies: if some columns vary much more than others, PCA without standardization will preferentially align with those larger-scale dimensions.

When `standardize=True`, CrossLearn recomputes all three pieces walk-forward:

- mean
- standard deviation
- PCA loadings

No future row is used when transforming the current row.

## Related APIs

- [Chronos guide](./chronos.md)
- `ChronosExtractor` for online embedding without adaptive PCA
- `ChronosEmbedder.transform_dataframe(...)` for lower-level aligned embedding augmentation

# Chronos in CrossLearn

This document explains how Chronos is integrated into CrossLearn, both as a user-facing feature and as an implementation detail.

CrossLearn exposes four Chronos-backed APIs:

- `ChronosExtractor` for online embedding inside a policy forward pass
- `WalkForwardChronosWrapper` for env-side online embedding when an environment emits one observation at a time
- `embed_dataframe` for the high-level offline dataframe-to-environment workflow
- `ChronosEmbedder` for direct window embedding and lower-level dataframe augmentation

All four are built around Chronos-2 as a reusable time-series representation model. CrossLearn also supports other Chronos families like Chronos-Bolt and Chronos T5, as well as custom Chronos-compatible pipelines. To learn more about Chronos and the supported model families, see the [Chronos documentation](https://github.com/amazon-science/chronos-forecasting).

For leakage-safe dimensionality reduction on Chronos embeddings, see
[walk-forward PCA workflows](./pca_workflows.md).

## Overview

The Chronos integration is designed around one idea: the reinforcement-learning policy should see a standard feature vector, while Chronos handles the time-series encoding behind the scenes.

- `ChronosExtractor` is the online path. It receives batched observations, converts them into rolling windows if needed, runs Chronos embeddings, pools the token-level outputs, and returns one feature vector per observation.
- `WalkForwardChronosWrapper` is the env-side online path. It collects single observations into a rolling or expanding history and returns Chronos vectors directly from the environment.
- `embed_dataframe` is the high-level offline path. It slices the requested dataframe history, runs Chronos embedding over rolling windows, trims the alignment warmup rows, and returns the aligned dataframe you hand to an offline environment.
- `ChronosEmbedder` is the lower-level utility underneath these paths. It takes windows directly or derives them from a dataframe, embeds them in batches, and can append aligned embedding columns back into the dataframe.

The APIs share the same normalization, feature-selection, pooling, and Chronos loading logic.

## Components

### `embed_dataframe`

`embed_dataframe` is the main offline convenience API.

It is responsible for:

- validating `lookback` and `frame_bound`
- slicing the exact dataframe history needed for the requested window range
- creating an internal `ChronosEmbedder`
- appending aligned Chronos embedding columns
- trimming the first `lookback - 1` warmup rows
- optionally dropping the resolved source features after the aligned `chronos_*` columns have been appended
- returning the trimmed dataframe with both the original columns and aligned `chronos_*` columns

Use `embed_dataframe` when you want a ready-to-wire offline dataframe for dataframe-backed environments such as `gym-anytrading`.

### `ChronosEmbedder`

`ChronosEmbedder` is the low-level utility. It is responsible for:

- loading the Chronos pipeline
- resolving `device_map`
- normalizing input windows into a batched `(batch, n_features, lookback)` tensor
- selecting a subset of features when configured
- using direct model encoding for recognized Amazon Chronos families, with
  `pipeline.embed(...)` as a compatibility fallback
- pooling token embeddings into one vector per window
- returning either `numpy.float32` arrays or torch tensors

Use `ChronosEmbedder` when you want direct control over embedding windows or when you want to augment a dataframe directly without the extra slicing and trimming contract of `embed_dataframe`.

### `ChronosExtractor`

`ChronosExtractor` is the policy-facing wrapper. It is a `BaseFeaturesExtractor` implementation, so it fits the same contract as the vector and CNN extractors in CrossLearn.

It is responsible for:

- validating the observation-space layout
- creating an internal `ChronosEmbedder`
- inferring the Chronos embedding size during initialization
- optionally projecting that embedding into a requested `features_dim`

Use `ChronosExtractor` when your environment observations are already rolling windows or flat legacy windows and you want Chronos to sit directly in the online policy path.

### `WalkForwardChronosWrapper`

`WalkForwardChronosWrapper` is the environment-facing online wrapper for environments that emit one observation at a time instead of a full rolling window.

It is responsible for:

- flattening each raw observation with Gymnasium's space utilities
- collecting observations into a history buffer
- returning a placeholder vector until `min_history` observations have been collected
- embedding either the last `lookback` observations or the full expanding history
- replacing the wrapped environment observation with one pooled Chronos vector

Use `WalkForwardChronosWrapper` when the environment is sequential but does not naturally emit `(lookback, n_features)` windows. In SB3 workflows, place it after any environment wrappers that normalize observations or adapt the env to SB3. The policy can then use a plain `MlpPolicy` because the environment already emits Chronos vectors.

Key options:

- `lookback`: number of recent observations used for each rolling Chronos window
- `min_history`: number of observations required before returning real embeddings; defaults to `lookback`
- `warmup_value`: placeholder value used before `min_history`
- `expanding_window`: if `True`, embed all retained history after warmup
- `max_history`: optional cap for expanding-window history
- `cache_size`: LRU cache size for repeated online windows

## Accepted Inputs

Both the extractor and the embedder accept the following observation shapes:

- `3D`: `(batch, lookback, n_features)` or `(batch, n_features, lookback)`
- `2D`: `(lookback, n_features)` for a single window
- `2D`: `(batch, lookback * n_features)` for batched flat legacy windows when `lookback` is provided
- `1D`: `(lookback * n_features,)` for a single flat legacy window when `lookback` is provided

The normalization path converts all of these into a batched `float32` tensor shaped `(batch, n_features, lookback)`. This matches the Chronos-2 API contract, where features are variates and lookback is the history length.

`ChronosExtractor` also infers the expected window layout from the Gymnasium observation space:

- a 2D observation space is treated as `(lookback, n_features)`
- a 1D observation space is treated as flat legacy input and therefore requires `lookback`

## Feature Selection

Chronos does not need to see every feature column if you only want a subset of the observation.

CrossLearn supports three related inputs:

- `feature_names`: names for the full feature dimension
- `selected_columns`: select features by name
- `selected_indices`: select features by integer position

Rules:

- use either `selected_columns` or `selected_indices`, not both
- `selected_columns` requires `feature_names`
- selected indices must be within the feature dimension of the window

Selection happens after shape normalization on the feature/variate axis and before the call into Chronos.

## Pooling

Chronos returns token-level embeddings. CrossLearn reduces them to one vector per window using one of two pooling modes:

- `mean`: average all token embeddings
- `last`: use the final token embedding

This pooled vector becomes the feature vector exposed to the policy or returned by the embedder.

## `device_map` and Actual Device Flow

`device_map` controls where the Chronos model is loaded. For recognized Amazon Chronos model families, CrossLearn moves only cache-miss windows to the model device for direct encoder inference. The generic `pipeline.embed(...)` fallback still uses Chronos' CPU-staged input path.

In CrossLearn, `ChronosExtractor` resolves `device_map` automatically from the agent device by default, so the model follows the agent device when possible. For the native agent path specifically:

- `device_map="auto"` is resolved to the best available torch device
- the agent forwards its resolved device into the Chronos extractor kwargs when the extractor supports `device_map`
- the Chronos pipeline is loaded using that resolved `device_map`

This means a CUDA-enabled agent will load the Chronos model on CUDA by default. GPU utilization still depends on batch size and cache-miss volume. To maximize Chronos throughput on GPU, use larger `n_envs` in your vectorized environment to create wider inference batches. Only enable async environment stepping if the environment latency is large enough to justify the process overhead.

### Why the Chronos input is still CPU-staged

Chronos' `pipeline.embed(...)` implementation batches inputs through an internal PyTorch `DataLoader` that uses a pin-memory path. That path expects dense CPU tensors. If a CUDA tensor is passed directly to `pipeline.embed(...)`, PyTorch raises an error like:

`RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned`

Because of that, CrossLearn stages the selected window tensor onto CPU before using the generic `pipeline.embed(...)` fallback. Direct encoder inference for recognized Amazon Chronos families bypasses this DataLoader path after cache lookup.

This is specific to the Chronos integration. It does not mean the entire agent or extractor path has to remain on CPU.

This design separation is intentional. CrossLearn keeps three concerns independent:

1. **Observation-shape normalization** - flattening, reshaping, and feature selection happen before embeddings
2. **Chronos model placement via `device_map`** - controls where the model lives independently
3. **Caller-visible output placement via `output_device`** - ensures outputs land where the rest of the policy expects them

This separation is what allows CrossLearn to respect Chronos' CPU input requirement for `pipeline.embed(...)` without changing the surrounding RL code. The Chronos model can still live on GPU, the input gets staged to CPU for cache lookup and generic embedding, and the output returns to GPU for the rest of the policy call without requiring Chronos-specific workarounds in your training loop.

## Online Path: `ChronosExtractor`

The online Chronos flow looks like this:

1. The agent converts the environment observation batch onto the agent device.
2. `ChronosExtractor.forward(...)` receives that batched tensor.
3. The embedder normalizes the window shape to `(batch, n_features, lookback)` and applies feature selection.
4. CrossLearn checks the LRU embedding cache for repeated selected windows.
5. Cache misses use direct encoder inference for recognized Amazon Chronos families, otherwise `pipeline.embed(...)`.
6. Custom Chronos-like pipelines remain supported through the generic fallback path.
7. CrossLearn pools the returned token embeddings into one vector per window.
8. The pooled features are moved onto `output_device`, which for the online path is the same device as the incoming observation tensor.
9. The extractor projection layer and the rest of the policy continue from there.

Important consequence:

- the Chronos model can still live on CUDA
- Chronos-2, Chronos-Bolt, and Chronos T5 use direct encoder inference for cache misses when available
- custom Chronos-compatible pipelines can still use the generic embedding path
- the pooled output can still be returned to CUDA for the rest of the policy

### Initialization-time embedding probe

`ChronosExtractor` performs one embedding call during initialization. This is used to infer the embedding width so the extractor can expose a stable `features_dim` and optionally build a projection layer.

That means Chronos integration errors can appear at extractor construction time, before the first training step.

## Env-Side Online Path: `WalkForwardChronosWrapper`

The env-side Chronos flow looks like this:

1. The wrapped environment emits one raw observation.
2. `WalkForwardChronosWrapper` flattens and stores that observation.
3. Until `min_history` observations are available, the wrapper returns a placeholder vector.
4. After warmup, the wrapper builds a rolling or expanding history matrix.
5. The embedder normalizes that matrix to `(batch, n_features, lookback)`, checks the cache, and runs Chronos only for misses.
6. The wrapper returns the pooled Chronos vector as the agent-visible observation.

For CityLearn with Stable-Baselines3, the typical wrapper order is:

```python
from stable_baselines3.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from crosslearn.envs import WalkForwardChronosWrapper

env = CityLearnEnv("citylearn_challenge_2023_phase_2_local_evaluation", central_agent=True)
env = NormalizedObservationWrapper(env)
env = StableBaselines3Wrapper(env)
env = WalkForwardChronosWrapper(
    env,
    lookback=4,
    min_history=4,
    model_name="amazon/chronos-bolt-tiny",
)

model = SAC("MlpPolicy", env, seed=42)
```

## Offline Path: `embed_dataframe` and `ChronosEmbedder`

The preferred offline entry point is `embed_dataframe(...)`.

Typical offline use cases:

- precompute Chronos features for dataframe-backed environments
- build an aligned feature dataframe for offline training
- embed one or more windows directly when you need lower-level control

When using `embed_dataframe(...)`:

1. CrossLearn validates `lookback` and `frame_bound`.
2. It slices `df.iloc[frame_bound[0] - lookback : frame_bound[1]]`.
3. It runs `ChronosEmbedder.transform_dataframe(...)` on that history slice.
4. It trims the first `lookback - 1` rows so the returned dataframe starts where the environment starts.
5. It returns the trimmed dataframe with aligned `chronos_*` columns.

Set `drop_feature_names=True` if you want the returned dataframe to remove
the resolved source features after adding the new `chronos_*` columns.

`ChronosEmbedder.transform_dataframe(...)` is the lower-level path.

When using `transform_dataframe(...)` directly:

1. CrossLearn resolves which columns to embed.
2. It creates rolling windows from the dataframe values.
3. It embeds those windows in batches.
4. It appends the resulting Chronos embedding columns back to a copy of the dataframe.
5. The first `lookback - 1` rows are filled with `NaN` so the embeddings stay aligned with the original index.

If `progress_bar=True`, CrossLearn displays a `tqdm` progress bar with estimated time remaining for the embedding process. This is useful for large dataframes (and long lookback periods) where embedding can take a while.

## Practical Examples

### Online REINFORCE with Chronos

```python
import numpy as np
from gym_anytrading.datasets import STOCKS_GOOGL
from gym_anytrading.envs import StocksEnv

from crosslearn import REINFORCE, make_vec_env
from crosslearn.extractors import ChronosExtractor

LOOKBACK = 30
FEATURE_NAMES = ["Open", "High", "Low", "Close", "Volume"]
SELECTED_COLUMNS = ["Close", "Volume"]
FRAME_BOUND = (LOOKBACK, len(STOCKS_GOOGL))


def online_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, "Close"].to_numpy()[start:end]
    signal_features = env.df.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32)[start:end]
    return prices, signal_features


class OnlineStocksEnv(StocksEnv):
    _process_data = online_process_data


vec_env = make_vec_env(
    lambda: OnlineStocksEnv(
        df=STOCKS_GOOGL,
        window_size=LOOKBACK,
        frame_bound=FRAME_BOUND,
    ),
    n_envs=4,
)

agent = REINFORCE(
    vec_env,
    features_extractor_class=ChronosExtractor,
    features_extractor_kwargs={
        "feature_names": ["Open", "High", "Low", "Close", "Volume"],
        "selected_columns": ["Close", "Volume"],
    },
    seed=42,
)
agent.learn(total_timesteps=100_000)
```

### Offline trading dataframe

```python
import numpy as np
from gym_anytrading.datasets import STOCKS_GOOGL
from gym_anytrading.envs import StocksEnv

from crosslearn.extractors import embed_dataframe

LOOKBACK = 30
FEATURE_NAMES = ["Open", "High", "Low", "Close", "Volume"]
SELECTED_COLUMNS = ["Close", "Volume"]
FRAME_BOUND = (LOOKBACK, len(STOCKS_GOOGL))

offline_df = embed_dataframe(
    STOCKS_GOOGL,
    lookback=LOOKBACK,
    frame_bound=FRAME_BOUND,
    feature_names=FEATURE_NAMES,
    selected_columns=SELECTED_COLUMNS,
    progress_bar=True,
)


class OfflineStocksEnv(StocksEnv):
    def __init__(self, prices, signal_features, **kwargs):
        self._prices = prices
        self._signal_features = signal_features.astype(np.float32)
        super().__init__(**kwargs)

    def _process_data(self):
        return self._prices, self._signal_features


offline_env = OfflineStocksEnv(
    prices=offline_df["Close"].to_numpy(dtype=np.float32),
    signal_features=offline_df.filter(like="chronos_").to_numpy(dtype=np.float32),
    df=offline_df,
    window_size=1,
    frame_bound=(1, len(offline_df)),
)
```

### Lower-level dataframe augmentation

```python
from crosslearn.extractors import ChronosEmbedder

embedder = ChronosEmbedder(
    feature_names=["Open", "Close", "Volume"],
    selected_columns=["Close", "Volume"],
)

transformed = embedder.transform_dataframe(
    df,
    lookback=30,
    columns=["Open", "Close", "Volume"],
    progress_bar=True,
)
```

## Troubleshooting

### `ImportError` for Chronos

Install the Chronos extras:

```bash
pip install "crosslearn[chronos]"
```

### `selected_columns requires feature_names`

When selecting by name, provide the full ordered list of feature names for the input window so CrossLearn can resolve names into indices.

### `lookback is required`

Flat legacy inputs do not encode their time dimension explicitly. Provide `lookback` (at **most** the value of `frame_bound[0]`) so CrossLearn can reconstruct `(lookback, n_features)`.

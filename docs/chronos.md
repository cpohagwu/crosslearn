# Chronos in CrossLearn

This document explains how Chronos is integrated into CrossLearn, both as a user-facing feature and as an implementation detail.

CrossLearn exposes two Chronos-backed components:

- `ChronosExtractor` for online embedding inside a policy forward pass
- `ChronosEmbedder` for offline embedding of rolling windows and dataframe augmentation

Both are built around Chronos-2 as a reusable time-series representation model, not as an implementation of ChronosRL.

## Overview

The Chronos integration is designed around one idea: the reinforcement-learning policy should see a standard feature vector, while Chronos handles the time-series encoding behind the scenes.

- `ChronosExtractor` is the online path. It receives batched observations, converts them into rolling windows if needed, runs Chronos embeddings, pools the token-level outputs, and returns one feature vector per observation.
- `ChronosEmbedder` is the offline path. It takes windows directly or derives them from a dataframe, embeds them in batches, and can append aligned embedding columns back into the dataframe.

The two paths share the same normalization, feature-selection, pooling, and Chronos loading logic.

## Components

### `ChronosEmbedder`

`ChronosEmbedder` is the low-level utility. It is responsible for:

- loading the Chronos pipeline
- resolving `device_map`
- normalizing input windows into a batched `(batch, lookback, n_features)` tensor
- selecting a subset of features when configured
- calling `pipeline.embed(...)`
- pooling token embeddings into one vector per window
- returning either `numpy.float32` arrays or torch tensors

Use `ChronosEmbedder` when you want direct control over embedding windows or when you want to build offline features for a dataframe.

### `ChronosExtractor`

`ChronosExtractor` is the policy-facing wrapper. It is a `BaseFeaturesExtractor` implementation, so it fits the same contract as the vector and CNN extractors in CrossLearn.

It is responsible for:

- validating the observation-space layout
- creating an internal `ChronosEmbedder`
- inferring the Chronos embedding size during initialization
- optionally projecting that embedding into a requested `features_dim`

Use `ChronosExtractor` when your environment observations are already rolling windows or flat legacy windows and you want Chronos to sit directly in the online policy path.

## Accepted Inputs

Both the extractor and the embedder accept the following observation shapes:

- `3D`: `(batch, lookback, n_features)`
- `2D`: `(lookback, n_features)` for a single window
- `2D`: `(batch, lookback * n_features)` for batched flat legacy windows when `lookback` is provided
- `1D`: `(lookback * n_features,)` for a single flat legacy window when `lookback` is provided

The normalization path converts all of these into a batched `float32` tensor shaped `(batch, lookback, n_features)`.

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

Selection happens after shape normalization and before the call into Chronos.

## Pooling

Chronos returns token-level embeddings. CrossLearn reduces them to one vector per window using one of two pooling modes:

- `mean`: average all token embeddings
- `last`: use the final token embedding

This pooled vector becomes the feature vector exposed to the policy or returned by the embedder.

## `device_map` and Actual Device Flow

`device_map` controls where the Chronos model is loaded, not the device of the tensor passed by the caller into `pipeline.embed(...)`.

In CrossLearn:

- `device_map="auto"` is resolved to the best available torch device
- for the native agent path, the agent forwards its resolved device into the Chronos extractor kwargs when the extractor supports `device_map`
- the Chronos pipeline is loaded using that resolved `device_map`

That means a CUDA-enabled agent will still load the Chronos model on CUDA by default.

### Why the Chronos input is still CPU-staged

Chronos' `pipeline.embed(...)` implementation batches inputs through an internal PyTorch `DataLoader` that uses a pin-memory path. That path expects dense CPU tensors. If a CUDA tensor is passed directly to `pipeline.embed(...)`, PyTorch raises an error like:

`RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned`

Because of that, CrossLearn always stages the input window tensor onto CPU immediately before calling `pipeline.embed(...)`.

This is specific to the Chronos integration. It does not mean the entire agent or extractor path has to remain on CPU.

## Online Path: `ChronosExtractor`

The online Chronos flow looks like this:

1. The agent converts the environment observation batch onto the agent device.
2. `ChronosExtractor.forward(...)` receives that batched tensor.
3. The embedder normalizes the window shape and applies feature selection.
4. CrossLearn stages the selected window batch onto CPU.
5. `pipeline.embed(...)` runs Chronos' internal batching and model inference.
6. CrossLearn pools the returned token embeddings into one vector per window.
7. The pooled features are moved onto `output_device`, which for the online path is the same device as the incoming observation tensor.
8. The extractor projection layer and the rest of the policy continue from there.

Important consequence:

- the Chronos model can still live on CUDA
- the Chronos input batch is CPU-staged before the `embed(...)` call
- the pooled output can still be returned to CUDA for the rest of the policy

### Initialization-time embedding probe

`ChronosExtractor` performs one embedding call during initialization. This is used to infer the embedding width so the extractor can expose a stable `features_dim` and optionally build a projection layer.

That means Chronos integration errors can appear at extractor construction time, before the first training step.

## Offline Path: `ChronosEmbedder`

`ChronosEmbedder` can also be used without an RL agent.

Typical offline use cases:

- embed one or more windows directly
- precompute Chronos features for training data
- append aligned embedding columns to a dataframe

When using `transform_dataframe(...)`:

1. CrossLearn resolves which columns to embed.
2. It creates rolling windows from the dataframe values.
3. It embeds those windows in batches.
4. It appends the resulting Chronos embedding columns back to a copy of the dataframe.
5. The first `lookback - 1` rows are filled with `NaN` so the embeddings stay aligned with the original index.

If `progress_bar=True`, batching is still used, but CrossLearn also displays a `tqdm` progress bar.

## Practical Examples

### Online REINFORCE with Chronos

```python
from crosslearn import REINFORCE, make_vec_env
from crosslearn.extractors import ChronosExtractor

vec_env = make_vec_env(lambda: MyTradingEnv(window_size=30), n_envs=4)

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

### Offline dataframe augmentation

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
    output_prefix="chronos_",
    progress_bar=True,
)
```

## Troubleshooting

### `ImportError` for Chronos

Install the Chronos extras:

```bash
pip install "crosslearn[chronos]"
```

### `RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned`

This happens when a CUDA tensor reaches `pipeline.embed(...)`. In CrossLearn, the fix is to CPU-stage the window tensor before the Chronos embed call while keeping the model placement controlled by `device_map`.

### `selected_columns requires feature_names`

When selecting by name, provide the full ordered list of feature names for the input window so CrossLearn can resolve names into indices.

### `lookback is required`

Flat legacy inputs do not encode their time dimension explicitly. Provide `lookback` so CrossLearn can reconstruct `(lookback, n_features)`.

## Design Notes

The Chronos integration tries to keep three concerns separate:

- observation-shape normalization
- Chronos model placement via `device_map`
- caller-visible output placement via `output_device`

That separation is what allows CrossLearn to respect Chronos' CPU input requirement without changing the surrounding RL code or introducing a Chronos-specific public API knob for the workaround.

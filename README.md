<div align="center">
  <a href="https://github.com/cpohagwu/crosslearn">
    <img
      src="https://github.com/cpohagwu/crosslearn/blob/main/docs/_static/crosslearn.png?raw=true"
      alt="crosslearn logo"
      width="900"
    >
  </a>
</div>

<div align="center">

# CrossLearn

**Reusable representation extractors for reinforcement learning.**

**CrossLearn** is an extractor-first reinforcement learning package built around a simple idea:
the most reusable part of many RL systems is not the algorithm, but the observation encoder.
The library keeps the algorithm surface intentionally small while giving vectors, images, and
time-series windows a shared feature-extraction interface that works with both native
`REINFORCE` and Stable-Baselines3.

</div>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-D22128?style=for-the-badge&logo=apache&logoColor=white)](https://github.com/cpohagwu/crosslearn/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/crosslearn?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=3775A9)](https://pypi.org/project/crosslearn/)
[![Downloads](https://img.shields.io/pypi/dm/crosslearn?style=for-the-badge&color=3775A9&label=Downloads)](https://pypi.org/project/crosslearn/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#installation)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-1f6feb?style=for-the-badge)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.3%2B-f59e0b?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Chronos](https://img.shields.io/badge/Chronos-Foundation%20Model-16a34a?style=for-the-badge)](https://github.com/amazon-science/chronos-forecasting)
[![Colab Quickstarts](https://img.shields.io/badge/Colab-6%20Quickstarts-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](#quickstart-colab-notebooks)

</div>

<div align="center">

[Why CrossLearn](#why-crosslearn) |
[Representation Families](#representation-families) |
[Quickstart Colab Notebooks](#quickstart-colab-notebooks) |
[Installation](#installation) |
[Core API](#core-api) |
[References](#references)

</div>

## Why CrossLearn

- **Extractor-first.** Representation learning is decoupled from agent algorithms. Build and reuse feature encoders across native REINFORCE and SB3.
- **Observation-agnostic interface.** Dense vectors, image stacks, and time-series windows inherit from `BaseFeaturesExtractor`, working with any SB3-compatible policy.
- **Chronos support.** Chronos-2 time-series encoder integrated for both online and offline workflows via `ChronosExtractor`, `embed_dataframe`, and walk-forward PCA utilities.
- **Minimal surface.** Agents remain lightweight; complexity lives in the extractor layer where it can be tested and reused independently.

## Representation Families

| Observation family | Components | Typical shape | Extractor |
| --- | --- | --- | --- |
| Flat vectors | Dense features | `(n_features,)` | `FlattenExtractor` |
| Images | Atari-style frames | `(C, H, W)` | `AtariPreprocessor` + `NatureCNNExtractor` |
| Time series | Rolling windows | `(lookback, n_features)` | `ChronosExtractor` or `embed_dataframe` |

All extractors implement SB3's `BaseFeaturesExtractor` interface, enabling reuse across native REINFORCE and Stable-Baselines3 policies.

## Chronos Workflows

Chronos-2 time-series encoding supports three core APIs plus a separate walk-forward PCA stage:

- `ChronosExtractor` - Online embedding within policy forward pass.
- `embed_dataframe` - Dataframe slicing and pre-embedding for offline training.
- `ChronosEmbedder` - Low-level embedding control for custom pipelines.
- `walkforward_pca_dataframe` - Leakage-safe walk-forward PCA for Chronos or generic numeric dataframes.

Features: configurable pooling (`mean` / `last`), feature selection, and automatic CUDA alignment with CPU-staged input.

See [Chronos implementation guide](https://github.com/cpohagwu/crosslearn/blob/main/docs/chronos.md) and [walk-forward PCA workflows](https://github.com/cpohagwu/crosslearn/blob/main/docs/pca_workflows.md) for details.

## Quickstart Colab Notebooks

| Notebook | Task | Environment |
| --- | --- | --- |
| [Native REINFORCE](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/01_cartpole-lunarlander_reinforce.ipynb) | Policy gradient baseline | CartPole-v1, LunarLander-v3 |
| [Atari REINFORCE + CNN](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/02_atari_reinforce_cnn.ipynb) | Image observations with NatureCNN | Atari |
| [Atari SB3 + CNN](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/03_atari_sb3_cnn.ipynb) | PPO with package extractor | Atari |
| [Chronos-2 + REINFORCE](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/04_gym-anytrading_reinforce_chronos2.ipynb) | Time-series encoder (online + offline) | Trading (OHLCV) |
| [Chronos-2 + SB3](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/05_gym-anytrading_sb3_chronos2.ipynb) | Time-series encoder (online + offline) | Trading (OHLCV) |
| [Chronos-2 + Walk-Forward PCA](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/06_gym-anytrading_reinforce_chronos2_walkforward_pca.ipynb) | Time-series encoder with adaptive PCA | Trading (OHLCV) |

## Installation

```bash
pip install crosslearn
pip install "crosslearn[atari]"
pip install "crosslearn[chronos]"
pip install "crosslearn[extra]"
```

`chronos` includes the Chronos foundation-model dependencies plus `tqdm` for offline embedding
progress bars. `extra` adds those dependencies alongside Atari support, TensorBoard, and
Weights & Biases. Notebook-only example dependencies are kept separate.

## Core API

Minimal native quickstart:

```python
from crosslearn import REINFORCE, make_vec_env

vec_env = make_vec_env("CartPole-v1", n_envs=4)
agent = REINFORCE(vec_env, seed=42)
agent.learn(total_timesteps=100_000)
```

See [the native REINFORCE guide](https://github.com/cpohagwu/crosslearn/blob/main/docs/reinforce.md)
for a full explanation of the native REINFORCE implementation, environment
handling, policy architecture, logging, and verbose training output.

Chronos-backed online trading quickstart:

```python
import numpy as np
from gym_anytrading.datasets import STOCKS_GOOGL
from gym_anytrading.envs import StocksEnv

from crosslearn import REINFORCE, make_vec_env
from crosslearn.extractors import ChronosExtractor

LOOKBACK = 30
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SELECTED_COLUMNS = ["Close", "Volume"]
FRAME_BOUND = (LOOKBACK, len(STOCKS_GOOGL))


def online_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, "Close"].to_numpy()[start:end]
    signal_features = env.df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)[start:end]
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

Chronos-backed offline trading quickstart:

```python
import numpy as np
from gym_anytrading.datasets import STOCKS_GOOGL
from gym_anytrading.envs import StocksEnv

from crosslearn import REINFORCE, make_vec_env
from crosslearn.extractors import embed_dataframe

LOOKBACK = 30
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SELECTED_COLUMNS = ["Close", "Volume"]
FRAME_BOUND = (LOOKBACK, len(STOCKS_GOOGL))

offline_df = embed_dataframe(
    STOCKS_GOOGL,
    lookback=LOOKBACK,
    frame_bound=FRAME_BOUND,
    feature_columns=FEATURE_COLUMNS,
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


def make_offline_env():
    return OfflineStocksEnv(
        prices=offline_df["Close"].to_numpy(dtype=np.float32),
        signal_features=offline_df.filter(like="chronos_").to_numpy(dtype=np.float32),
        df=offline_df,
        window_size=1,
        frame_bound=(1, len(offline_df)),
    )


offline_agent = REINFORCE(make_vec_env(make_offline_env, n_envs=1), seed=42)
offline_agent.learn(total_timesteps=100_000)
```

## Also Included

- `make_vec_env` normalizes string env IDs, single `gym.Env` instances, vector envs, and
  callable factories into a consistent `gym.vector.VectorEnv`.
- Callback utilities include solved-threshold stopping, checkpointing, best-model hooks,
  early stopping, and a progress bar.
- Logging integrations include TensorBoard and Weights & Biases run/config handling.

## Research Context
**CrossLearn** provides a simple and practical way to bring powerful pretrained models into reinforcement learning. Instead of building complex new algorithms, the library focuses on the representation layer - the part that turns raw observations into useful features for the policy.

The design is deliberately straightforward: inherit from `BaseFeaturesExtractor` and implement a `forward` method. The resulting extractor works seamlessly with both the package’s native REINFORCE agent and Stable-Baselines3 policies. This makes it easy to experiment with different observation types without rewriting the training loop.

A key example is using Chronos, a pretrained time-series model, to create richer features from rolling windows of data (such as financial OHLCV). Rather than treating time-series as a niche case, `crosslearn` treats pretrained encoders as interchangeable backbones.
The same approach extends naturally to image observations with stronger CNNs, multimodal models, or custom representation pipelines. By keeping the extractor layer reusable and decoupled from the agent, `crosslearn` enables faster experimentation and more effective learning across vector, image, and sequential data.

## References

- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.*
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.*
- [Ansari et al. (2024), *Chronos: Learning the Language of Time Series*](https://openreview.net/forum?id=gerNCVqqtR)
- [Ansari et al. (2025), *Chronos-2: From Univariate to Universal Forecasting*](https://arxiv.org/abs/2510.15821)
- [Lima, Oliveira, and Zanchettin (2025), *ChronosRL: embeddings-based reinforcement learning agent for financial trading*](https://doi.org/10.1016/j.procs.2025.07.132)

## Development and Contributions
- CrossLearn is actively being developed on [GitHub](https://github.com/cpohagwu/crosslearn). Please note that the API is subject to change as the library evolves, but we welcome contributions and feedback.
- If you encounter any issues, have suggestions for improvements, or want to contribute new features, please open an [issue](https://github.com/cpohagwu/crosslearn/issues) or submit a [pull request](https://github.com/cpohagwu/crosslearn/pulls) on the GitHub repository.

## License

**CrossLearn** is released under the Apache License 2.0. See the
[LICENSE](https://github.com/cpohagwu/crosslearn/blob/main/LICENSE).

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
[![Downloads](https://img.shields.io/pypi/dm/crosslearn?style=for-the-badge&logo=pypi&logoColor=white&label=Downloads&color=3775A9)](https://pypistats.org/packages/crosslearn)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#installation)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-1f6feb?style=for-the-badge)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.3%2B-f59e0b?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Chronos](https://img.shields.io/badge/Chronos-Foundation%20Model-16a34a?style=for-the-badge)](https://github.com/amazon-science/chronos-forecasting)
[![Colab Quickstarts](https://img.shields.io/badge/Colab-5%20Quickstarts-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](#quickstart-colab-notebooks)

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

- **Extractor-first design.** The package is organized around reusable extractors that let you plug flat features, image encoders, or foundation-model embeddings into the same RL training interface.
- **One interface for native and SB3 training.** The same extractor classes can power the
  package's native `REINFORCE` policy or be passed into SB3 through `policy_kwargs`.
- **Foundation-model time-series support is first-class.** `ChronosExtractor` and
  `ChronosEmbedder` let pretrained Chronos models act as RL backbones for rolling windows.
- **Observation families stay consistent.** Flat vectors, Atari-style image stacks, and
  multivariate time-series windows all plug into the same feature-extraction contract.
- **The extension path is clean.** If you can encode an observation batch into feature
  vectors, you can turn that encoder into a reusable RL backbone.

## Representation Families

| Observation family | Main components | Typical observation shape | Why it matters |
| --- | --- | --- | --- |
| Flat vectors | `FlattenExtractor` | `(n_features,)` | Keeps classic control and tabular-style numeric tasks simple and fast. |
| Images | `AtariPreprocessor` + `NatureCNNExtractor` | `(C, H, W)` | Gives Atari-style grayscale, resized, stacked frames with a standard CNN backbone. |
| Rolling time series | `ChronosExtractor` or `ChronosEmbedder` | `(lookback, n_features)` or flat legacy windows | Lets a pretrained time-series foundation model serve as the observation encoder. |

All packaged extractors implement the SB3 `BaseFeaturesExtractor` contract, which is the key
reason the same backbone can move between native `crosslearn.REINFORCE` and SB3 policies.

## Chronos Workflows

**CrossLearn** includes two complementary Chronos paths:

- `ChronosExtractor` embeds rolling windows online inside the policy forward pass.
- `ChronosEmbedder` embeds windows offline and writes aligned embedding columns back into a
  dataframe.

They both use the Chronos-2 multivariate forecasting model, which gives you a powerful pretrained time-series encoder without needing to train your own sequence model from scratch.

The Chronos utilities are designed for practical RL use:

- They accept raw 2D rolling windows or flat backward-compatible inputs.
- They support feature selection by `selected_columns` or `selected_indices`.
- They expose `mean` and `last` pooling over Chronos token embeddings.
- They align with CUDA automatically by default when available, so the online
  Chronos path does not bounce observations through CPU unless you override
  `device_map`.

## Quickstart Colab Notebooks

Checkout the Colab quickstarts for runnable examples of native and SB3 training with vector, image, and time-series observations:

| Notebook | Focus | Colab |
| --- | --- | --- |
| Native REINFORCE quickstart | Shortest path from `make_vec_env` to a working policy-gradient baseline on `CartPole-v1` or `LunarLander-v3`. | [Open in Colab](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/01_cartpole-lunarlander_reinforce.ipynb) |
| Atari REINFORCE with Nature CNN | Native Atari training with `AtariPreprocessor` and `NatureCNNExtractor`. | [Open in Colab](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/02_atari_reinforce_cnn.ipynb) |
| Atari PPO with the package CNN extractor | SB3 `PPO` using the same image backbone interface. | [Open in Colab](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/03_atari_sb3_cnn.ipynb) |
| Chronos-2 trading features with native REINFORCE | Online and offline Chronos workflows over rolling OHLCV windows. | [Open in Colab](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/04_gym-anytrading_reinforce_chronos2.ipynb) |
| Chronos-2 trading features with SB3 PPO | The same Chronos representation path plugged into SB3. | [Open in Colab](https://colab.research.google.com/github/cpohagwu/crosslearn/blob/main/examples/05_gym-anytrading_sb3_chronos2.ipynb) |

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

```python
from crosslearn import REINFORCE, make_vec_env
from crosslearn.envs import AtariPreprocessor
from crosslearn.extractors import (
    BaseFeaturesExtractor,
    ChronosEmbedder,
    ChronosExtractor,
    FlattenExtractor,
    NatureCNNExtractor,
)
```

Minimal native quickstart:

```python
from crosslearn import REINFORCE, make_vec_env

vec_env = make_vec_env("CartPole-v1", n_envs=4)
agent = REINFORCE(vec_env, seed=42)
agent.learn(total_timesteps=100_000)
```

Chronos-backed time-series quickstart:

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

`ChronosExtractor` follows the agent device automatically by default. That removes the
largest avoidable CPU/GPU transfer in the online Chronos path, but GPU utilization still
depends on batch size: use larger `n_envs` than the minimal notebook demos when you want
wider Chronos inference batches, and only enable async env stepping when environment latency
is large enough to justify process overhead.

SB3 interoperability with the same extractor contract:

```python
import gymnasium as gym
from stable_baselines3 import PPO

from crosslearn.envs import AtariPreprocessor
from crosslearn.extractors import NatureCNNExtractor

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
env = AtariPreprocessor(env, stack_size=4, frame_skip=1, screen_size=84)

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs={"features_extractor_class": NatureCNNExtractor},
    verbose=1,
)
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

## License

**CrossLearn** is released under the Apache License 2.0. See the
[LICENSE](https://github.com/cpohagwu/crosslearn/blob/main/LICENSE).

# Native REINFORCE in CrossLearn

This document explains how the native `REINFORCE` agent in CrossLearn works, both as a user-facing training API and as an implementation detail.

CrossLearn's native agent is intentionally narrow:

- one on-policy policy-gradient algorithm
- one shared policy class with a swappable feature extractor
- one vectorized training loop that works across classic control, image, and time-series environments

The result is a simple baseline that stays compatible with different observation encoders without changing the training loop.

## Overview

CrossLearn's `REINFORCE` implementation is a vectorized Williams-style policy-gradient agent.

At a high level, each update does this:

1. collect `n_steps` transitions from each of `n_envs` environments
2. compute discounted returns across the collected rollout
3. flatten the rollout into one large batch
4. run one policy update over the whole batch

Total transitions per update are:

`n_steps * n_envs`

This matters when you tune throughput and reporting. For example, `n_steps=512` and `n_envs=4` means every update is built from 2048 transitions.

### What this means in practice

The easiest way to misread this is to think the agent waits for one environment to advance 2048 local steps before updating. That is not what happens.

With `n_steps=512` and `n_envs=4`:

- each environment advances 512 local steps during the rollout
- the vectorized batch contributes 2048 total transitions to the update
- the policy update runs after those 512 parallel vector steps, not after 2048 local steps in one environment

So if you have a trading environment with 2330 rows and one episode spans those 2330 rows, then the first few updates look like this:

1. update 1 collects local steps `0 -> 512` from each of the 4 environments
2. update 2 collects local steps `512 -> 1024` from each environment
3. update 3 collects local steps `1024 -> 1536` from each environment
4. update 4 collects local steps `1536 -> 2048` from each environment
5. update 5 is the first rollout where those environments can actually hit the episode end around local step 2330

The printed global step count is still cumulative across all environments, so by the time update 5 finishes the agent will report `5 * 512 * 4 = 10,240` timesteps even though each individual environment has only advanced about 2330 local steps.

## Environment Handling

All native agents operate on a `gym.vector.VectorEnv` internally, even if the caller passes something else.

CrossLearn normalizes the environment input through `make_vec_env(...)`.

### Supported environment inputs

#### Existing `gym.vector.VectorEnv`

If you pass a vectorized environment directly, it is used as-is.

- `n_envs` is ignored
- the agent trusts the existing vectorization layout

#### String Gymnasium env ID

If you pass a string like `"CartPole-v1"`, CrossLearn creates `n_envs` copies with `gym.make(...)`.

This is the simplest option for standard registered environments.

#### Callable factory

If you pass a callable like `lambda: MyTradingEnv(df, window_size=30)`, CrossLearn calls it `n_envs` times and wraps the results in a vector environment.

This is the preferred path for:

- custom environments
- parameterized environments
- environments that do not have a registered Gymnasium spec

#### Single `gym.Env` instance

If you pass a single environment instance, CrossLearn does not train on that exact object. Instead:

- it reads `env.spec.id`
- closes the provided instance
- recreates `n_envs` fresh copies from the registered spec

If the environment instance has no registered spec, CrossLearn raises an error and asks you to use a callable factory instead.

### Sync vs async vectorization

When the agent constructs the vector environment itself, it uses:

- `SyncVectorEnv` by default
- `AsyncVectorEnv` when `use_async_env=True`

General rule:

- use sync for cheap environments like CartPole or time-series backtests
- use async for slower environments where multi-process stepping can hide environment-side latency

For cheap environments, async can be slower because process overhead dominates.

## Observation and Action Assumptions

The native policy always sees batched observations shaped like:

`(n_envs, *obs_shape)` during rollout collection

and:

`(n_steps * n_envs, *obs_shape)` during the update step

The action space is currently assumed to be discrete. The shared policy builds a categorical distribution from action logits, so the native `REINFORCE` path is not a continuous-action implementation.

## Policy Architecture

The native agent uses `ActorCriticPolicy`, even though the REINFORCE loss only uses the actor side.

Architecture:

1. observation batch
2. feature extractor
3. shared MLP
4. actor head producing action logits
5. critic head producing scalar values

The critic head exists for architectural reuse. It is built and returned by the policy, but REINFORCE does not use critic values in its loss.

### Feature extractor

The default extractor is `FlattenExtractor`, which turns flat observations into a feature vector suitable for an MLP.

You can swap in any compatible `BaseFeaturesExtractor`, including:

- `FlattenExtractor`
- `NatureCNNExtractor`
- `ChronosExtractor`
- your own custom extractor

This is the main extension point in the native training stack.

### Shared MLP

After feature extraction, the policy builds a shared MLP with:

- default `net_arch=[64, 64]`
- default activation `nn.Tanh`

If `net_arch` is empty, the shared network becomes an identity mapping and the actor and critic heads operate directly on the extractor output.

### Actor and critic heads

The policy creates:

- one linear actor head of shape `shared_dim -> n_actions`
- one linear critic head of shape `shared_dim -> 1`

Action selection is:

- stochastic during training rollouts
- greedy when `predict(..., deterministic=True)` is used
- greedy during evaluation in `learn(eval_env=...)`

## Rollout Collection

Each rollout update is collected sequentially over time and in parallel across environments.

That means:

- there is a Python loop over the time axis
- there is no per-environment Python loop for policy evaluation

For each time step:

1. the last observation batch is converted from NumPy to `float32` torch on the agent device
2. the policy produces a categorical action distribution for the full batch
3. actions are sampled for all environments at once
4. the vector environment steps once with that action batch
5. rewards and done flags are stored in the rollout buffer
6. episode returns are updated for any environments that finished on that step

Both `terminated` and `truncated` are treated as terminal boundaries in the rollout buffer. There is no critic bootstrap term in this implementation.

### Important consequence for long episodes

If an episode is longer than `n_steps`, the episode spans multiple policy updates.

In the current implementation, returns are computed only over the collected rollout window for that update. There is no bootstrap term carried across rollout boundaries. So when an episode continues past the end of a rollout:

- training still proceeds normally
- the update still uses the collected rewards from that rollout
- but the return targets for that update are truncated at the rollout boundary rather than using the eventual full-episode return

## Return Computation

After collection, CrossLearn computes discounted returns backward over the time axis.

Important details:

- return computation loops over time in reverse
- the environment axis is handled by NumPy broadcasting
- when an environment is done at a given step, the running return for that environment is reset before the next earlier step is processed

If `normalize_returns=True`, the full rollout's returns are z-score normalized before the update step. This reduces gradient variance and acts like a simple baseline-style stabilization.

## Policy Update

After returns are computed, the rollout buffer is flattened from:

`(n_steps, n_envs, ...)`

to:

`(n_steps * n_envs, ...)`

The update then uses one batched call to `policy.evaluate_actions(...)`.

The current objective is:

- `policy_loss = -(log_probs * returns).mean()`
- `entropy_loss = -entropy_coeff * entropy.mean()`
- `total_loss = policy_loss + entropy_loss`

Then CrossLearn:

1. zeroes gradients
2. backpropagates `total_loss`
3. clips global gradient norm with `max_grad_norm`
4. steps the optimizer
5. steps the LR scheduler if one was configured

Default optimizer behavior:

- optimizer: `torch.optim.Adam`
- learning rate: `3e-4`

## Training Lifecycle

The native `learn(...)` loop supports either stop condition:

- `total_timesteps`
- `total_episodes`

or both at once.

Important counting rules:

- one timestep means one transition across all parallel environments
- `total_timesteps` is total environment interactions, not per-environment timesteps
- `total_episodes` is the total number of completed episodes summed across all vectorized environments

If `reset_num_timesteps=True`, internal counters and rolling reward state are reset before training starts. If `False`, training continues from the previous counters.

## Console Verbosity

CrossLearn has three console verbosity levels.

### `verbose=0`

No standard progress output from the agent.

### `verbose=1`

The agent prints:

- a one-line agent summary after construction
- a training-start line when `learn(...)` begins
- evaluation summaries when `eval_env` is provided
- callback stop notices
- a final completion line

Typical constructor summary fields:

- algorithm name
- `n_envs`
- resolved device
- observation shape
- action count

### `verbose=2`

This is the highest built-in console verbosity.

In addition to the `verbose=1` output, it prints:

- the full policy representation after construction
- one per-update training line after every gradient step

The per-update line reports:

- `update`: update count
- `steps`: cumulative total timesteps
- `episodes`: cumulative completed episodes
- `ep_reward`: mean reward of episodes that completed during the most recent rollout
- `rolling_mean_ep_reward`: rolling mean over the current reward window
- `loss`: current `train/total_loss`

### Important nuance about the printed rewards

`ep_reward` can be `0.0` for a rollout if no episode finished during that update window.

This is common in long-horizon environments. For example, if one episode in your trading environment lasts 2330 local steps and you train with `n_steps=512`, then:

- updates 1 through 4 will usually print `ep_reward=0.0`
- update 5 is the first one that can include completed episodes and therefore the first one that can print a nonzero `ep_reward`

This happens because `ep_reward` is not the mean reward over all transitions in the rollout. It is the mean return of only those episodes that actually finished during that rollout.

`rolling_mean_ep_reward` also stays `0.0` until the rolling reward deque is full:

- default rolling window: 100 episodes
- overridden by `EpisodeSolvedCallback(n_episodes=...)` when that callback is active

So early training output can show a real loss while the rolling reward still prints as zero.

## Logged Metrics

The console output is a compact summary. If you attach a logger, CrossLearn records a larger metric set.

Per update, the native REINFORCE path logs:

- `train/policy_loss`
- `train/entropy_loss`
- `train/total_loss`
- `train/entropy`
- `train/mean_episode_reward`
- `train/learning_rate`
- `train/n_timesteps`
- `train/n_episodes`
- `train/n_updates`
- `train/rolling_mean_episode_reward`

If `eval_env` is provided, it also logs:

- `eval/mean_reward`

### Evaluation behavior

Evaluation is always greedy:

- `predict(..., deterministic=True)` is used internally
- evaluation runs in a separate single environment
- mean reward is averaged over `n_eval_episodes`

If a new best eval mean reward is found, `on_best_model(...)` callbacks are triggered.

## Callbacks and Their Effect on Training

Callbacks run inside the native training loop and can stop training or save artifacts.

Notable effects:

- `EpisodeSolvedCallback` changes the rolling reward window used for the solved criterion
- `BestModelCallback` saves checkpoints on new best performance
- `CheckpointCallback` saves periodic checkpoints by update count
- `EarlyStoppingCallback` can stop training after stalled best-model events
- `ProgressBarCallback` shows a `tqdm` progress bar with `ep_reward`, `loss`, and `n_eps`

Callbacks do not change the policy architecture or loss directly, but they do affect when training stops and what gets saved or displayed.

## Prediction, Saving, and Loading

### Prediction

`predict(...)` accepts either:

- a single observation shaped like `obs_shape`
- a batch shaped like `(n, *obs_shape)`

It converts the input to `float32`, moves it to the agent device, and returns NumPy actions.

Default prediction is deterministic. For stochastic sampling at inference time, pass `deterministic=False`.

### Saving

`save(...)` stores:

- policy state
- optimizer state
- serialized hyperparameters
- timestep and episode counters
- best recorded eval mean reward

### Loading

`load(...)` restores the saved training state onto the current device.

`load_from_path(...)` constructs a new agent and immediately loads the checkpoint.

## Practical Examples

### Standard Gymnasium environment

```python
from crosslearn import REINFORCE, make_vec_env

vec_env = make_vec_env("CartPole-v1", n_envs=4)

agent = REINFORCE(
    vec_env,
    n_steps=512,
    normalize_returns=True,
    verbose=2,
    seed=42,
)
agent.learn(total_timesteps=100_000)
```

### Custom environment factory

```python
from crosslearn import REINFORCE

agent = REINFORCE(
    lambda: MyTradingEnv(df, window_size=30),
    n_envs=8,
    n_steps=256,
    use_async_env=False,
    seed=42,
)
agent.learn(total_timesteps=500_000)
```

### Custom feature extractor

```python
import torch.nn as nn

from crosslearn import REINFORCE

agent = REINFORCE(
    "CartPole-v1",
    features_extractor_class=MyExtractor,
    features_extractor_kwargs={"features_dim": 128},
    policy_kwargs={
        "net_arch": [128, 128],
        "activation_fn": nn.ReLU,
    },
)
```

## Troubleshooting

### `Provide total_timesteps or total_episodes (or both).`

`learn(...)` requires at least one stopping condition.

### `Cannot vectorize a gym.Env that has no registered spec.`

You passed a single environment instance that cannot be recreated from a Gymnasium ID. Wrap it in a callable factory instead.

Example:

```python
agent = REINFORCE(lambda: MyEnv(...), n_envs=4)
```

### Rewards look like zero in the console early in training

That usually means one of these is true:

- no episode finished during the latest rollout, so `ep_reward` is `0.0`
- the rolling reward window is not full yet, so `rolling_mean_ep_reward` is still `0.0`

This does not necessarily mean the loss or action distribution is broken.

### Evaluation behaves differently from training

That is expected. Training rollouts sample from the categorical policy, while evaluation uses greedy action selection.

### Continuous action spaces are not supported

The native policy builds a categorical distribution over discrete actions. Use the native REINFORCE path only with discrete-action environments.

## Design Notes

The native REINFORCE implementation is deliberately simple:

- vectorized environment handling
- one batched policy update per rollout
- extractor-first observation processing
- minimal algorithm surface

That simplicity is why the same native training loop can power flat vectors, image stacks, and Chronos-backed time-series windows without rewriting the agent logic.

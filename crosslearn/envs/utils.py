from __future__ import annotations

from typing import Callable, Union

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


def make_vec_env(
    env_input: Union[str, gym.Env, gym.vector.VectorEnv, Callable],
    n_envs: int = 1,
    use_async: bool = False,
) -> gym.vector.VectorEnv:
    """
    Convert any environment specification into a vectorized environment.

    Agents in this package operate on a ``VectorEnv`` internally so the
    policy receives batched observations ``(n_envs, *obs_shape)``.

    Args:
        env_input:
            - ``gym.vector.VectorEnv`` → returned unchanged; ``n_envs`` is ignored.
            - ``gym.Env`` instance → closed and recreated ``n_envs`` times from
              its registered spec.  Raises if the env has no spec (e.g. custom
              envs without registration) — pass a callable factory instead.
            - ``str`` → treated as a Gymnasium env ID.
            - ``Callable`` → called ``n_envs`` times; use for custom or
              parametrised envs: ``lambda: MyTradingEnv(df, ...)``.
        n_envs: Number of parallel environments. Ignored for VectorEnv input.
        use_async: Use ``AsyncVectorEnv`` (multi-process) instead of
            ``SyncVectorEnv`` (single-process).  Async is faster for slow envs
            (physics, real-time data).  For cheap envs (CartPole, time series)
            the process overhead makes sync faster. Default: ``False``.

    Returns:
        A ``gym.vector.VectorEnv`` with ``num_envs == n_envs``.

    Examples::

        # Already vectorised — used as-is
        vec = gym.make_vec("CartPole-v1", num_envs=4)
        vec = make_vec_env(vec)

        # String ID
        vec = make_vec_env("CartPole-v1", n_envs=4)

        # Custom env factory (required for envs without a registered spec)
        vec = make_vec_env(lambda: MyTradingEnv(df, window_size=20), n_envs=4)
    """
    # ── Already vectorised ───────────────────────────────────────────────
    if isinstance(env_input, gym.vector.VectorEnv):
        return env_input

    VecCls = AsyncVectorEnv if use_async else SyncVectorEnv

    # ── String env ID ────────────────────────────────────────────────────
    if isinstance(env_input, str):
        env_id = env_input
        return VecCls([lambda eid=env_id: gym.make(eid) for _ in range(n_envs)])

    # ── Callable factory ─────────────────────────────────────────────────
    if callable(env_input) and not isinstance(env_input, gym.Env):
        return VecCls([env_input for _ in range(n_envs)])

    # ── Single gym.Env instance ──────────────────────────────────────────
    if isinstance(env_input, gym.Env):
        spec = env_input.spec
        if spec is None:
            raise ValueError(
                "Cannot vectorize a gym.Env that has no registered spec. "
                "Wrap it in a callable factory instead:\n"
                "  make_vec_env(lambda: MyEnv(...), n_envs=4)"
            )
        env_id = spec.id
        env_input.close()
        return VecCls([lambda eid=env_id: gym.make(eid) for _ in range(n_envs)])

    raise TypeError(
        f"env_input must be a str, gym.Env, gym.vector.VectorEnv, or callable. "
        f"Got {type(env_input).__name__}."
    )

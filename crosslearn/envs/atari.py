from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing

try:
    from gymnasium.wrappers import FrameStackObservation
except ImportError:
    from gymnasium.experimental.wrappers import (
        FrameStackObservationV0 as FrameStackObservation,
    )


class AtariPreprocessor(
    gym.Wrapper[np.ndarray, int, np.ndarray, int],
    gym.utils.RecordConstructorArgs,
):
    """
    Apply Atari preprocessing and channels-first frame stacking.

    This wrapper is built on Gymnasium's ``AtariPreprocessing`` and always:

    - converts observations to grayscale
    - resizes frames to ``screen_size x screen_size``
    - leaves observations as ``uint8`` in ``[0, 255]``
    - does not clip rewards
    - stacks frames as ``(stack_size, screen_size, screen_size)``

    Rewards are passed through unchanged when ``frame_skip=1``. If
    ``frame_skip > 1``, the reward is the sum across repeated action steps,
    which is standard frame-skip behavior and not reward clipping.

    For ALE environments, create the base env with ``frameskip=1`` if you set
    ``frame_skip > 1`` here to avoid double frame skipping.

    Args:
        env: Atari env or another env compatible with ``AtariPreprocessing``.
        stack_size: Number of recent grayscale frames to stack on the leading
            observation axis. Default: ``4``.
        noop_max: Maximum number of no-op actions sampled on reset.
            Default: ``30``.
        frame_skip: Number of repeated action steps inside the preprocessor.
            Default: ``1`` to preserve raw per-step rewards.
        screen_size: Output frame width and height. Default: ``84``.
        terminal_on_life_loss: If ``True``, a life loss ends the episode from
            the agent's perspective. Default: ``False``.

    Example::

        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
        env = AtariPreprocessor(env, stack_size=4, frame_skip=1, screen_size=84)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        stack_size: int = 4,
        noop_max: int = 30,
        frame_skip: int = 1,
        screen_size: int = 84,
        terminal_on_life_loss: bool = False,
    ) -> None:
        gym.utils.RecordConstructorArgs.__init__(
            self,
            stack_size=stack_size,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
        )

        wrapped_env = AtariPreprocessing(
            env,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        wrapped_env = FrameStackObservation(wrapped_env, stack_size=stack_size)

        super().__init__(wrapped_env)


__all__ = ["AtariPreprocessor"]

from __future__ import annotations

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from crosslearn.extractors.base import BaseFeaturesExtractor


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Default extractor for flat (1-D) vector observations.

    Simply flattens the observation. No learnable parameters — acts as a
    pass-through so the actor/critic MLP does all the work.

    Compatible with any ``gym.spaces.Box`` with ``ndim == 1``.

    Args:
        observation_space: Must be a flat ``Box`` space.
        features_dim: Ignored; inferred from ``observation_space``.

    Example::

        extractor = FlattenExtractor(env.single_observation_space)
        obs = torch.zeros(8, 4)   # batch of 8 CartPole observations
        out = extractor(obs)      # (8, 4)
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        fd = int(np.prod(observation_space.shape))
        super().__init__(observation_space, fd)
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations.float())
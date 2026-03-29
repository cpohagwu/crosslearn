from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor as _SB3Base


class BaseFeaturesExtractor(_SB3Base, ABC):
    """Thin SB3-backed base class for package extractors."""

    def __init__(self, observation_space: gym.Space, features_dim: int) -> None:
        super().__init__(observation_space, features_dim)

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observations into feature vectors."""

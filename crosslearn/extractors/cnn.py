from __future__ import annotations

from typing import Type

import gymnasium as gym
import torch
import torch.nn as nn

from crosslearn.extractors.base import BaseFeaturesExtractor


class NatureCNNExtractor(BaseFeaturesExtractor):
    """
    Convolutional feature extractor following the architecture from
    Mnih et al. (2015) — the original DQN paper ("Nature CNN").

    Expects images in **channels-first** format ``(C, H, W)``. For Atari-style
    training, pair this extractor with ``crosslearn.envs.AtariPreprocessor``
    so the policy sees grayscale, resized, frame-stacked observations like
    ``(4, 84, 84)``.

    Args:
        observation_space: A ``Box`` space of shape ``(C, H, W)``.
        features_dim: Dimension of the output feature vector. Default: 512.
        activation_fn: Activation between conv layers. Default: ``nn.ReLU``.

    Example::

        from crosslearn.envs import AtariPreprocessor

        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
        env = AtariPreprocessor(env, stack_size=4, frame_skip=1, screen_size=84)
        extractor = NatureCNNExtractor(env.observation_space, features_dim=512)
        obs = torch.zeros(4, 4, 84, 84)     # (batch, C, H, W)
        out = extractor(obs)                # (4, 512)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, (
            f"NatureCNNExtractor expects image obs (C, H, W). "
            f"Got shape {observation_space.shape}."
        )
        c, h, w = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), activation_fn(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), activation_fn(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), activation_fn(),
            nn.Flatten(),
        )
        # Compute flattened CNN output dim with a dry run
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, c, h, w)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            activation_fn(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        if x.numel() > 0 and x.detach().amax().item() > 1.0:
            x = x / 255.0
        return self.linear(self.cnn(x))

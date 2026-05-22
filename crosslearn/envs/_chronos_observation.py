from __future__ import annotations

from typing import Any, Literal, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.spaces import utils as space_utils


ObservationMode = Literal["auto", "window", "stream"]
ResolvedObservationMode = Literal["window", "stream"]


class ChronosObservationAdapter:
    """Normalize env observations into Chronos windows.

    ``feature_names`` always names the per-timestep feature axis. For window
    observations that is the second dimension of ``(lookback, n_features)``;
    for stream observations it is the flattened single-step observation.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        lookback: int,
        observation_mode: ObservationMode = "auto",
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.observation_space = observation_space
        self.lookback = int(lookback)
        if self.lookback <= 0:
            raise ValueError("lookback must be greater than 0.")

        if observation_mode not in {"auto", "window", "stream"}:
            raise ValueError(
                "observation_mode must be one of 'auto', 'window', or 'stream'."
            )

        self.mode = self._resolve_mode(observation_mode)
        self.n_features = self._resolve_n_features(feature_names)
        self.feature_names = list(feature_names) if feature_names is not None else None
        if self.feature_names is not None and len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries, but "
                f"the per-timestep feature axis has {self.n_features} features."
            )

    def _space_shape(self) -> tuple[int, ...] | None:
        shape = getattr(self.observation_space, "shape", None)
        if shape is None:
            return None
        return tuple(int(dim) for dim in shape)

    def _resolve_mode(self, observation_mode: ObservationMode) -> ResolvedObservationMode:
        if observation_mode != "auto":
            return observation_mode

        shape = self._space_shape()
        if shape is not None and len(shape) == 2 and shape[0] == self.lookback:
            return "window"
        return "stream"

    def _resolve_n_features(self, feature_names: Sequence[str] | None) -> int:
        shape = self._space_shape()
        if self.mode == "window":
            if shape is not None and len(shape) == 2 and shape[0] == self.lookback:
                return int(shape[1])
            flatdim = int(space_utils.flatdim(self.observation_space))
            if feature_names is not None:
                n_features = len(feature_names)
                if flatdim != self.lookback * n_features:
                    raise ValueError(
                        "Window observation flat size does not match "
                        f"lookback * len(feature_names): {flatdim} != "
                        f"{self.lookback} * {n_features}."
                    )
                return n_features
            if flatdim % self.lookback != 0:
                raise ValueError(
                    "Cannot infer n_features for flat window observations. "
                    "Provide feature_names whose length matches the per-timestep "
                    "feature axis."
                )
            return flatdim // self.lookback

        return int(space_utils.flatdim(self.observation_space))

    def flatten_stream_observation(self, observation: Any) -> np.ndarray:
        try:
            flattened = space_utils.flatten(self.observation_space, observation)
        except Exception:
            flattened = np.asarray(observation, dtype=np.float32).reshape(-1)
        flattened = np.asarray(flattened, dtype=np.float32).reshape(-1)
        if flattened.size != self.n_features:
            raise ValueError(
                f"Expected stream observation with {self.n_features} features, "
                f"got {flattened.size}."
            )
        return flattened

    def window_from_observation(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape == (self.lookback, self.n_features):
            return array.copy()

        flattened = array.reshape(-1)
        expected = self.lookback * self.n_features
        if flattened.size != expected:
            try:
                flattened = space_utils.flatten(self.observation_space, observation)
            except Exception:
                pass
            flattened = np.asarray(flattened, dtype=np.float32).reshape(-1)

        if flattened.size != expected:
            raise ValueError(
                f"Expected window observation with shape "
                f"({self.lookback}, {self.n_features}) or flat size {expected}, "
                f"got flat size {flattened.size}."
            )
        return flattened.reshape(self.lookback, self.n_features).copy()


__all__ = ["ChronosObservationAdapter", "ObservationMode", "ResolvedObservationMode"]

from crosslearn.agents.reinforce import REINFORCE
from crosslearn.envs import AtariPreprocessor, WalkForwardChronosPCAWrapper
from crosslearn.envs.utils import make_vec_env
from crosslearn.registry import list_agents, make_agent
from crosslearn.callbacks import (
    BaseCallback,
    BestModelCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    EpisodeSolvedCallback,
    ProgressBarCallback,
)
from crosslearn.loggers import BaseLogger, TensorBoardLogger, WandbLogger
from crosslearn.extractors import (
    BaseFeaturesExtractor,
    ChronosEmbedder,
    ChronosExtractor,
    FlattenExtractor,
    NatureCNNExtractor,
)

__version__ = "0.3.15"

__all__ = [
    "REINFORCE",
    "make_vec_env",
    "AtariPreprocessor",
    "WalkForwardChronosPCAWrapper",
    "make_agent",
    "list_agents",
    "BaseCallback",
    "BestModelCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "EpisodeSolvedCallback",
    "ProgressBarCallback",
    "BaseLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "BaseFeaturesExtractor",
    "FlattenExtractor",
    "NatureCNNExtractor",
    "ChronosEmbedder",
    "ChronosExtractor",
]

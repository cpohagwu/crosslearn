from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseLogger(ABC):
    """
    Abstract logger interface.

    ``log()`` is required.  ``log_artifact()`` and ``close()`` are no-ops
    for backends that do not support them — callers never need to guard
    these calls.
    """

    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a dict of scalar metrics at ``step``."""
        ...

    def log_artifact(
        self, path: str, name: str, artifact_type: str = "model"
    ) -> None:
        """Upload a file as a versioned artifact (no-op if unsupported)."""

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log a run configuration snapshot (no-op if unsupported)."""

    def close(self) -> None:
        """Flush and finalise. Called automatically at the end of ``learn()``."""


class TensorBoardLogger(BaseLogger):
    """
    Log scalar metrics to TensorBoard.

    A **datetime-stamped subfolder** is always created under ``log_dir`` so
    that successive runs do not overwrite each other.  The resolved path is
    printed at construction time.

    Args:
        log_dir: Root directory for TensorBoard event files. Default: ``"./tb_logs"``.
        run_name: Optional human-readable run label.  The final folder is
            ``{log_dir}/{run_name}_{YYYYMMDD_HHMMSS}/``.
            If ``None``, only the timestamp is used.
        n_envs: Number of parallel environments.  Stored as a hyperparameter
            in the TensorBoard hparams tab.

    Example::

        logger = TensorBoardLogger("./logs", run_name="reinforce_cartpole", n_envs=4)
        # tensorboard --logdir=./logs
    """

    def __init__(
        self,
        log_dir: str = "./tb_logs",
        run_name: Optional[str] = None,
        n_envs: int = 1,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"{run_name}_{ts}" if run_name else ts
        self._run_dir = os.path.join(log_dir, folder)
        self.writer = SummaryWriter(self._run_dir)
        self.n_envs = n_envs
        self._config_logged = False

        print(f"  [TensorBoardLogger] Writing to {self._run_dir}")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                self.writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                pass

    def close(self) -> None:
        self.writer.close()

    def __repr__(self) -> str:
        return f"TensorBoardLogger(run_dir='{self._run_dir}')"

    def log_config(self, config: Dict[str, Any]) -> None:
        if self._config_logged:
            return
        self._config_logged = True

        run_dir = Path(self._run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        pretty = json.dumps(config, indent=2, sort_keys=True)
        config_path.write_text(pretty, encoding="utf-8")

        self.writer.add_text("config/json", pretty, global_step=0)

        flat = _flatten_config(config)
        hparams = {}
        for key, value in flat.items():
            if value is None:
                hparams[key] = "None"
            elif isinstance(value, (str, int, float, bool)):
                hparams[key] = value
            else:
                hparams[key] = json.dumps(value, sort_keys=True)
        self.writer.add_hparams(hparams, {})


class WandbLogger(BaseLogger):
    """
    Log metrics and model artifacts to Weights & Biases.

    Args:
        project: W&B project name (required).
        name: Optional run name.
        config: Dict of hyperparameters to associate with the run.
            Pass ``agent._get_hyperparams()`` for automatic population.
        tags: List of string tags.
        n_envs: Number of parallel environments, added to ``config`` automatically.

    Example::

        logger = WandbLogger(
            project="crosslearn",
            name="reinforce_breakout",
            config={"gamma": 0.999, "n_steps": 2048},
            n_envs=4,
        )
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        n_envs: int = 1,
    ) -> None:
        import wandb

        cfg = config or {}
        cfg["n_envs"] = n_envs
        wandb.init(project=project, name=name, config=cfg, tags=tags or [])
        self._wandb = wandb
        self._config_logged = False

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        self._wandb.log(
            {k: v for k, v in metrics.items() if v is not None}, step=step
        )

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        art = self._wandb.Artifact(name=name, type=artifact_type)
        art.add_file(path)
        self._wandb.log_artifact(art)

    def close(self) -> None:
        self._wandb.finish()

    def __repr__(self) -> str:
        return f"WandbLogger(project='{self._wandb.run.project}')"

    def log_config(self, config: Dict[str, Any]) -> None:
        if self._config_logged:
            return
        self._config_logged = True
        self._wandb.config.update(
            {"training_config": config}, allow_val_change=True
        )


def _flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in config.items():
        flat_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            items.update(_flatten_config(value, flat_key))
        else:
            items[flat_key] = value
    return items

"""Weights & Biases sink (optional dependency).

This is intentionally isolated so the rest of the codebase never needs to
import `wandb` directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .sink import TelemetrySink

_logger = logging.getLogger(__name__)


def _flatten(metrics: Dict[str, float], namespace: Optional[str]) -> Dict[str, float]:
    if not namespace:
        return metrics
    return {f"{namespace}/{k}": v for k, v in metrics.items()}


@dataclass
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: Optional[str] = None  # "online" | "offline" | "disabled"
    log_tables: bool = True
    log_files: bool = True


class WandbSink(TelemetrySink):
    def __init__(self, cfg: WandbConfig, *, config_dict: Optional[Dict] = None):
        self._cfg = cfg
        self._wandb = None
        self._run = None
        self._config_dict = config_dict or {}

        if not cfg.enabled:
            return

        try:
            import wandb  # type: ignore
        except Exception as exc:
            _logger.warning("W&B enabled but wandb import failed: %s", exc)
            return

        self._wandb = wandb
        init_kwargs = {
            "project": cfg.project,
            "entity": cfg.entity,
            "name": cfg.run_name,
            "tags": cfg.tags,
            "config": self._config_dict,
        }
        if cfg.mode:
            init_kwargs["mode"] = cfg.mode
        # Drop None values so wandb doesn't complain.
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        self._run = wandb.init(**init_kwargs)

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log_metrics(
        self,
        metrics: Dict[str, float],
        *,
        step: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        payload = _flatten(metrics, namespace)
        self._wandb.log(payload, step=step)

    def log_text(self, name: str, text: str) -> None:
        if not self.enabled:
            return
        # Use a 1-row table so it renders nicely in the UI.
        table = self._wandb.Table(columns=["text"], data=[[text]])
        self._wandb.log({name: table})

    def log_table(self, name: str, rows: List[Dict]) -> None:
        if not (self.enabled and self._cfg.log_tables):
            return
        if not rows:
            return
        columns = sorted({k for r in rows for k in r.keys()})
        data = [[r.get(c) for c in columns] for r in rows]
        table = self._wandb.Table(columns=columns, data=data)
        self._wandb.log({name: table})

    def log_files(self, name: str, paths: Iterable[str]) -> None:
        if not (self.enabled and self._cfg.log_files):
            return
        # Minimal approach: attach files to the run directory via wandb.save.
        # (Artifacts are possible too, but this keeps the adapter lightweight.)
        saved = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                continue
            try:
                self._wandb.save(str(path))
                saved.append(str(path))
            except Exception as exc:
                _logger.warning("Failed to save file to W&B (%s): %s", path, exc)
        if saved:
            self._wandb.log({f"{name}/count": float(len(saved))})

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._wandb.finish()
        finally:
            self._run = None


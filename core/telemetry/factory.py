from __future__ import annotations

import logging
from typing import Optional

from .sink import NullSink, TelemetrySink
from .wandb_sink import WandbConfig, WandbSink

_logger = logging.getLogger(__name__)


def build_sink(cfg) -> TelemetrySink:
    """Build a telemetry sink from a Config-like object.

    Expects cfg.wandb.* to exist; if it doesn't, returns NullSink.
    """
    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None:
        return NullSink()

    enabled = bool(getattr(wandb_cfg, "enabled", False))
    if not enabled:
        return NullSink()

    wc = WandbConfig(
        enabled=True,
        project=getattr(wandb_cfg, "project", None),
        entity=getattr(wandb_cfg, "entity", None),
        run_name=getattr(wandb_cfg, "run_name", None),
        tags=list(getattr(wandb_cfg, "tags", []) or []) or None,
        mode=getattr(wandb_cfg, "mode", None),
        log_tables=bool(getattr(wandb_cfg, "log_tables", True)),
        log_files=bool(getattr(wandb_cfg, "log_files", True)),
    )

    # Log config as plain dict when available (keeps W&B UI useful).
    config_dict: Optional[dict] = None
    if hasattr(cfg, "to_dict"):
        try:
            config_dict = cfg.to_dict()
        except Exception:
            config_dict = None

    sink = WandbSink(wc, config_dict=config_dict)
    if not getattr(sink, "enabled", False):
        _logger.warning("W&B was enabled but sink failed to initialize; continuing without telemetry.")
        return NullSink()
    return sink


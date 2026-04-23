"""Telemetry sink interface (pluggable backends).

This module defines a minimal contract for logging metrics/tables/files without
coupling the training/eval code to a specific vendor (e.g. Weights & Biases).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class TelemetrySink(Protocol):
    def log_metrics(
        self,
        metrics: Dict[str, float],
        *,
        step: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> None: ...

    def log_text(self, name: str, text: str) -> None: ...

    def log_table(self, name: str, rows: List[Dict]) -> None: ...

    def log_files(self, name: str, paths: Iterable[str]) -> None: ...

    def finish(self) -> None: ...


@dataclass
class NullSink:
    """Default sink — does nothing."""

    def log_metrics(
        self,
        metrics: Dict[str, float],
        *,
        step: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> None:
        return None

    def log_text(self, name: str, text: str) -> None:
        return None

    def log_table(self, name: str, rows: List[Dict]) -> None:
        return None

    def log_files(self, name: str, paths: Iterable[str]) -> None:
        return None

    def finish(self) -> None:
        return None


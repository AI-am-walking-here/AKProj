"""Small in-memory store for extracted metrics/rows.

Useful for building tables and logging the same structured data to multiple
destinations (CSV/MD/TEX, W&B Table, JSON, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricStore:
    rows: List[Dict] = field(default_factory=list)

    def add_row(self, row: Dict) -> None:
        self.rows.append(dict(row))


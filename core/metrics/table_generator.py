"""Results table generator — outputs CSV, Markdown, and LaTeX.

Builds comparison tables aligned with DINOv3 Table 4 layout:
  Model | Backbone | AP | AP50 | AP75 | AP_S | AP_M | AP_L

Each row is a dict containing model metadata and metric values.
All AP values are displayed ×100 (e.g. 0.412 → 41.2) with one decimal place.
The LaTeX output uses booktabs and bolds the best value per numeric column.
"""
import csv
import logging
from pathlib import Path
from typing import Dict, List

_logger = logging.getLogger(__name__)

# Columns in display order, matching DINOv3 Table 4.
_META_COLS = ["model_name", "backbone_type"]
_METRIC_COLS = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
_ALL_COLS = _META_COLS + _METRIC_COLS

# Human-readable headers for each column.
_HEADERS = {
    "model_name":    "Model",
    "backbone_type": "Backbone",
    "AP":            "AP",
    "AP50":          "AP50",
    "AP75":          "AP75",
    "AP_small":      "AP$_S$",
    "AP_medium":     "AP$_M$",
    "AP_large":      "AP$_L$",
}


def _scale(value: float) -> float:
    """Convert [0, 1] AP to display-scale (×100)."""
    return round(value * 100, 1)


def _best_per_col(rows: List[Dict]) -> Dict[str, float]:
    """Return the highest display-scale value per numeric column."""
    best: Dict[str, float] = {}
    for col in _METRIC_COLS:
        vals = [_scale(r[col]) for r in rows if col in r]
        if vals:
            best[col] = max(vals)
    return best


def build_results_table(
    rows: List[Dict],
    output_path: str,
    formats: List[str],
) -> Dict[str, str]:
    """Write comparison table in one or more formats.

    Args:
        rows: Each element is a dict that must contain:
            ``model_name``, ``backbone_type``, and the 6 COCO AP keys
            (``AP``, ``AP50``, ``AP75``, ``AP_small``, ``AP_medium``,
            ``AP_large``).  Extra keys (e.g. ``AR@1``) are ignored.
        output_path: Base path without extension, e.g.
            ``"results/table"``.  The function appends ``.csv``,
            ``.md``, and/or ``.tex`` as appropriate.
        formats: Subset of ``["csv", "markdown", "latex"]``.

    Returns:
        ``{format_name: file_path}`` for each format written.
    """
    if not rows:
        raise ValueError("rows is empty — nothing to write")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    if "csv" in formats:
        path = output_path.with_suffix(".csv")
        _write_csv(rows, path)
        written["csv"] = str(path)
        _logger.info(f"CSV table written to {path}")

    if "markdown" in formats:
        path = output_path.with_suffix(".md")
        _write_markdown(rows, path)
        written["markdown"] = str(path)
        _logger.info(f"Markdown table written to {path}")

    if "latex" in formats:
        path = output_path.with_suffix(".tex")
        _write_latex(rows, path)
        written["latex"] = str(path)
        _logger.info(f"LaTeX table written to {path}")

    return written


# --------------------------------------------------------------------------- #
# Format writers (each is a pure function — no shared mutable state)
# --------------------------------------------------------------------------- #

def _write_csv(rows: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            scaled = {k: v for k, v in row.items() if k in _META_COLS}
            for col in _METRIC_COLS:
                scaled[col] = _scale(row.get(col, 0.0))
            writer.writerow(scaled)


def _write_markdown(rows: List[Dict], path: Path) -> None:
    headers = [_HEADERS[c] for c in _ALL_COLS]
    sep = ["---"] * len(_ALL_COLS)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    for row in rows:
        cells = []
        for col in _META_COLS:
            cells.append(str(row.get(col, "")))
        for col in _METRIC_COLS:
            cells.append(f"{_scale(row.get(col, 0.0)):.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(rows: List[Dict], path: Path) -> None:
    best = _best_per_col(rows)
    num_cols = len(_ALL_COLS)
    col_spec = "ll" + "r" * len(_METRIC_COLS)  # left-align meta, right-align numbers

    def _fmt_cell(col: str, row: Dict) -> str:
        if col in _META_COLS:
            return str(row.get(col, ""))
        val = _scale(row.get(col, 0.0))
        s = f"{val:.1f}"
        if best.get(col) == val:
            s = r"\textbf{" + s + "}"
        return s

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Object detection results on COCO val2017.}",
        r"  \label{tab:detection}",
        r"  \begin{tabular}{" + col_spec + "}",
        r"    \toprule",
        "    " + " & ".join(_HEADERS[c] for c in _ALL_COLS) + r" \\",
        r"    \midrule",
    ]

    for row in rows:
        cells = [_fmt_cell(c, row) for c in _ALL_COLS]
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

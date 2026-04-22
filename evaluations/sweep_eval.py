"""Multi-model evaluation sweep — runs evaluate.py logic across all configs.

Reads a sweep YAML (see configs/sweep.yaml), evaluates each model, and
combines results into a single comparison table in CSV, Markdown, and LaTeX.
The LaTeX output is ready to paste into your paper (booktabs, best values bold).

Usage
-----
python evaluations/sweep_eval.py --sweep configs/sweep.yaml --output results/

Optional overrides:
python evaluations/sweep_eval.py --sweep configs/sweep.yaml --output results/ \\
    --per-class \\
    --formats csv markdown latex \\
    --batch-size 16 \\
    --device cuda
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from core.telemetry import MetricStore, build_sink

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sweep config schema
# --------------------------------------------------------------------------- #

@dataclass
class ModelSweepEntry:
    """One model's evaluation specification."""

    name: str
    config: str
    checkpoint: str
    ann_file: str
    img_dir: str
    output: str
    backbone_type: str = "vit"
    dataset: str = "coco"
    train_ann: Optional[str] = None
    batch_size: int = 8
    num_workers: int = 4
    per_class: bool = False


_REQUIRED_FIELDS = {"name", "config", "checkpoint", "ann_file", "img_dir", "output"}


def load_sweep_config(path: str) -> tuple:
    """Parse a sweep YAML into a list of ModelSweepEntry objects."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    sweep_defaults = raw.get("defaults", {})
    default_batch = sweep_defaults.get("batch_size", 8)
    default_workers = sweep_defaults.get("num_workers", 4)
    default_per_class = sweep_defaults.get("per_class", False)
    formats = sweep_defaults.get("formats", ["csv", "markdown", "latex"])
    table_output = sweep_defaults.get("table_output", "results/comparison_table")

    entries: List[ModelSweepEntry] = []
    for i, model_raw in enumerate(raw.get("models", [])):
        missing = _REQUIRED_FIELDS - set(model_raw.keys())
        if missing:
            raise ValueError(
                f"Sweep entry {i} (name={model_raw.get('name', '?')!r}) "
                f"is missing required fields: {missing}"
            )
        entries.append(
            ModelSweepEntry(
                name=model_raw["name"],
                config=model_raw["config"],
                checkpoint=model_raw["checkpoint"],
                ann_file=model_raw["ann_file"],
                img_dir=model_raw["img_dir"],
                output=model_raw["output"],
                backbone_type=model_raw.get("backbone_type", "vit"),
                dataset=model_raw.get("dataset", "coco"),
                train_ann=model_raw.get("train_ann", None),
                batch_size=model_raw.get("batch_size", default_batch),
                num_workers=model_raw.get("num_workers", default_workers),
                per_class=model_raw.get("per_class", default_per_class),
            )
        )

    _logger.info(f"Loaded {len(entries)} model(s) from sweep config {path}")
    return entries, sweep_defaults, table_output, formats


# --------------------------------------------------------------------------- #
# Sweep runner
# --------------------------------------------------------------------------- #

def run_sweep(
    entries: List[ModelSweepEntry],
    output_dir: str,
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    per_class_override: Optional[bool] = None,
    sink=None,
) -> List[Dict]:
    """Evaluate each model in *entries* and return list of result dicts."""
    import torch

    from evaluations.evaluate import (
        load_model_from_checkpoint,
        build_eval_dataloader,
        build_eval_mapping,
        run_full_evaluation,
        save_results,
        print_summary,
    )
    from core.config import Config, _deep_merge

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results: List[Dict] = []
    store = MetricStore()

    for idx, entry in enumerate(entries):
        _logger.info(f"\n{'='*60}")
        _logger.info(f"[{idx+1}/{len(entries)}] Evaluating: {entry.name}")
        _logger.info(f"{'='*60}")

        try:
            # Build config
            defaults = {
                "backbone": {
                    "type": "vit",
                    "name": "vit_base_patch16_rope_reg1_gap_256",
                    "checkpoint": None,
                    "pretrained": False,
                    "img_size": 256,
                },
                "head": {
                    "type": "detr",
                    "d_model": 256,
                    "nhead": 8,
                    "num_decoder_layers": 6,
                    "dim_feedforward": 2048,
                    "dropout": 0.1,
                    "num_queries": 100,
                    "aux_loss": True,
                },
                "loss": {"cls_type": "focal"},
                "eval": {"score_threshold": 0.01, "max_detections": 100},
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            with open(entry.config) as f:
                yaml_cfg = yaml.safe_load(f)
            _deep_merge(defaults, yaml_cfg)

            defaults["backbone"]["type"] = entry.backbone_type
            if device_override:
                defaults["device"] = device_override

            cfg = Config(defaults)
            device = torch.device(cfg.device)

            run_sink = sink or build_sink(cfg)

            # Class mapping
            source_to_target, target_label_to_cat_id, num_classes = build_eval_mapping(
                train_ann=entry.train_ann,
                eval_ann=entry.ann_file,
            )

            # Model
            model = load_model_from_checkpoint(
                checkpoint_path=entry.checkpoint,
                cfg=cfg,
                num_classes=num_classes,
                device=device,
            )

            # DataLoader
            bs = batch_size_override or entry.batch_size
            loader = build_eval_dataloader(
                img_dir=entry.img_dir,
                ann_file=entry.ann_file,
                cfg=cfg,
                batch_size=bs,
                num_workers=entry.num_workers,
            )

            # Evaluate
            per_class = per_class_override if per_class_override is not None else entry.per_class
            metrics = run_full_evaluation(
                model=model,
                loader=loader,
                ann_file=entry.ann_file,
                cfg=cfg,
                source_to_target=source_to_target,
                target_label_to_cat_id=target_label_to_cat_id,
                per_class=per_class,
                device=device,
            )

            # Save individual result
            out_path = Path(output_dir) / Path(entry.output).name
            save_results(
                metrics=metrics,
                model_name=entry.name,
                backbone_type=entry.backbone_type,
                checkpoint_path=entry.checkpoint,
                dataset=entry.dataset,
                ann_file=entry.ann_file,
                output_path=str(out_path),
            )
            print_summary(metrics, entry.name)

            # Collect for table
            row = {
                "model_name": entry.name,
                "backbone_type": entry.backbone_type,
            }
            row.update({k: v for k, v in metrics.items() if k != "per_class_ap"})
            all_results.append(row)
            store.add_row(row)

            # Per-model scalars
            run_sink.log_metrics(
                {k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
                namespace=entry.name,
            )

        except Exception as exc:
            _logger.error(f"FAILED: {entry.name} — {exc}", exc_info=True)
            continue

    return all_results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-model evaluation sweep with combined results table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sweep", required=True,
                   help="Path to sweep YAML config (e.g. configs/sweep.yaml)")
    p.add_argument("--output", default="results/",
                   help="Output directory for individual JSON results and the combined table")
    p.add_argument("--per-class", action="store_true", default=None,
                   help="Override per_class flag for all models")
    p.add_argument("--formats", nargs="+",
                   choices=["csv", "markdown", "latex"],
                   default=None,
                   help="Override output table formats from sweep YAML")
    p.add_argument("--table-output", default=None,
                   help="Override base path for the combined table (no extension)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size for all models")
    p.add_argument("--device", default=None,
                   help="Override device for all models (cuda/cpu)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    entries, sweep_defaults, table_output, formats = load_sweep_config(args.sweep)

    if args.formats is not None:
        formats = args.formats
    if args.table_output is not None:
        table_output = args.table_output

    # Place the combined table in the output directory if table_output is relative
    table_output_path = Path(args.output) / Path(table_output).name

    # Build one sink for the sweep run (uses YAML wandb: if present).
    # Reuses it for all entries to keep runs consolidated.
    cfg_for_sink = None
    try:
        with open(entries[0].config) as f:
            cfg_for_sink = yaml.safe_load(f) or {}
    except Exception:
        cfg_for_sink = {}

    defaults = {"wandb": {"enabled": False}}
    if isinstance(cfg_for_sink, dict):
        defaults.update(cfg_for_sink)
    from core.config import Config as _Cfg
    sweep_sink = build_sink(_Cfg(defaults))

    rows = run_sweep(
        entries=entries,
        output_dir=args.output,
        device_override=args.device,
        batch_size_override=args.batch_size,
        per_class_override=args.per_class,
        sink=sweep_sink,
    )

    if not rows:
        _logger.error("No models evaluated successfully — no table written")
        sys.exit(1)

    from core.metrics import build_results_table

    written = build_results_table(
        rows=rows,
        output_path=str(table_output_path),
        formats=formats,
    )

    sweep_sink.log_table("comparison_table", rows)
    sweep_sink.log_files("table_files", written.values())
    sweep_sink.finish()

    _logger.info("\nCombined table files:")
    for fmt, path in written.items():
        _logger.info(f"  {fmt:10s}: {path}")


if __name__ == "__main__":
    main()


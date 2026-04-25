"""Training schedule runner — CLI entrypoint.

Runs a multi-stage training schedule defined in YAML. Each stage calls
train.run_training in-process with a compiled config. Outputs go under
`output_root/<stage_name>/`; a combined rollup is written to
`output_root/schedule_results.json`.

Usage:
    # Full schedule
    python schedule.py --schedule configs/schedules/dinov3_style.yaml

    # Sanity-check only (1 epoch, 20 batches/stage, eval at epoch 1)
    python schedule.py --schedule configs/schedules/smoke_test.yaml --dry-run

    # Re-run a single stage (doesn't resume @prev unless the prior stage
    # already has last_epoch.pth on disk)
    python schedule.py --schedule configs/schedules/dinov3_style.yaml --only main

    # Resume from a specific stage (picks up prior stage's last_epoch.pth if it exists)
    python schedule.py --schedule configs/schedules/dinov3_style.yaml --start-from highres_adapt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.schedule import StageOverrides, run_schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a multi-stage training schedule.")
    p.add_argument("--schedule", type=Path, required=True, help="Path to schedule YAML.")
    p.add_argument("--only", type=str, default=None, help="Run a single named stage.")
    p.add_argument("--start-from", type=str, default=None,
                   help="Start from this stage, skipping earlier ones.")
    p.add_argument("--dry-run", action="store_true",
                   help="Preset smoke mode: 1 epoch, 20 batches, batch_size<=2, eval at epoch 1.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs per stage (auto-adjusts at_epochs to fit).")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap training samples seen per epoch (converted to batches internally).")
    p.add_argument("--eval-every-epoch", action="store_true",
                   help="Force eval at every training epoch (overrides YAML at_epochs).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.only and args.start_from:
        _logger.error("--only and --start-from are mutually exclusive")
        return 2

    if not args.schedule.exists():
        _logger.error(f"Schedule not found: {args.schedule}")
        return 2

    overrides = StageOverrides(
        dry_run=args.dry_run,
        epochs=args.epochs,
        max_samples_per_epoch=args.max_samples,
        force_eval_every_epoch=args.eval_every_epoch,
    )

    results = run_schedule(
        args.schedule,
        only=args.only,
        start_from=args.start_from,
        overrides=overrides,
    )

    print("\n" + "=" * 70)
    print(f"SCHEDULE COMPLETE ({len(results)} stage(s))")
    print("=" * 70)
    for r in results:
        print(f"  {r.name:30s}  status={r.status:8s}  best_ap={r.best_ap:.4f}  -> {r.output_dir}")

    failed = [r for r in results if r.status != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

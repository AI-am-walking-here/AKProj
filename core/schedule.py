"""Training schedule orchestrator.

Runs a sequence of training stages defined in YAML. Each stage:
    - Extends a base_config with stage-specific overrides
    - Optionally inits head weights from the previous stage's checkpoint (@prev)
    - Runs training via train.run_training (in-process)
    - Evaluates on one or more named suites at specified epochs

Design intent: decouple "what to run" (this module) from "how to run one training"
(train.run_training). Swap either independently.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.config import Config, _deep_merge, load_yaml

_logger = logging.getLogger(__name__)

PREV_TOKEN = "@prev"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvalDataset:
    name: str
    img_dir: str
    ann_file: str


@dataclass(frozen=True)
class EvalSpec:
    suites: List[str]                      # suite names referenced by stage
    at_epochs: List[int]                   # epochs to eval at (-1 = final)


@dataclass(frozen=True)
class TrainDatasetSpec:
    img_dir: str
    ann_file: str


@dataclass(frozen=True)
class Stage:
    name: str
    enabled: bool
    init_from: Optional[str]               # None | "@prev" | path
    train: TrainDatasetSpec
    overrides: Dict[str, Any]
    eval: EvalSpec


@dataclass(frozen=True)
class Schedule:
    name: str
    base_config: Path
    output_root: Path
    seed: Optional[int]
    eval_suites: Dict[str, List[EvalDataset]]
    stages: List[Stage]
    common_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    name: str
    status: str
    best_ap: float
    final_ckpt: Optional[str]
    output_dir: str
    summary_path: str


# --------------------------------------------------------------------------- #
# Schedule loader
# --------------------------------------------------------------------------- #

def load_schedule(path: Path) -> Schedule:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Schedule YAML must be a mapping, got {type(raw).__name__}")

    name = raw.get("name") or Path(path).stem
    base_config = Path(raw["base_config"])
    output_root = Path(raw["output_root"])
    seed = raw.get("seed")

    suites_raw = raw.get("eval_suites") or {}
    eval_suites: Dict[str, List[EvalDataset]] = {}
    for suite_name, items in suites_raw.items():
        datasets: List[EvalDataset] = []
        for it in items:
            datasets.append(EvalDataset(
                name=it.get("name") or suite_name,
                img_dir=it["img_dir"],
                ann_file=it["ann_file"],
            ))
        eval_suites[suite_name] = datasets

    stages_raw = raw.get("stages") or []
    stages: List[Stage] = []
    for s in stages_raw:
        train = s["train"]
        eval_cfg = s.get("eval") or {}
        stages.append(Stage(
            name=s["name"],
            enabled=bool(s.get("enabled", True)),
            init_from=s.get("init_from"),
            train=TrainDatasetSpec(img_dir=train["img_dir"], ann_file=train["ann_file"]),
            overrides=s.get("overrides") or {},
            eval=EvalSpec(
                suites=list(eval_cfg.get("suites") or []),
                at_epochs=list(eval_cfg.get("at_epochs") or []),
            ),
        ))

    if not stages:
        raise ValueError("Schedule has no stages")

    common_overrides = raw.get("common_overrides") or {}
    if not isinstance(common_overrides, dict):
        raise ValueError(
            f"common_overrides must be a mapping (same shape as stage.overrides), "
            f"got {type(common_overrides).__name__}"
        )

    return Schedule(
        name=name,
        base_config=base_config,
        output_root=output_root,
        seed=seed,
        eval_suites=eval_suites,
        stages=stages,
        common_overrides=common_overrides,
    )


# --------------------------------------------------------------------------- #
# Per-stage config compilation
# --------------------------------------------------------------------------- #

def _resolve_init_from(token: Optional[str], prev_ckpt: Optional[Path], stage_name: str) -> Optional[str]:
    if token is None:
        return None
    if token == PREV_TOKEN:
        if prev_ckpt is None:
            raise ValueError(
                f"Stage '{stage_name}' uses init_from: '@prev' but no previous stage produced a checkpoint"
            )
        return str(prev_ckpt)
    return token  # explicit path


def _suite_refs_to_eval_datasets(
    suite_names: List[str],
    suites: Dict[str, List[EvalDataset]],
    stage_name: str,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set = set()
    for suite_name in suite_names:
        if suite_name not in suites:
            raise ValueError(f"Stage '{stage_name}' references unknown eval_suite '{suite_name}'")
        for ed in suites[suite_name]:
            if ed.name in seen:
                continue  # dedupe
            seen.add(ed.name)
            out.append({"name": ed.name, "img_dir": ed.img_dir, "ann_file": ed.ann_file})
    return out


@dataclass(frozen=True)
class StageOverrides:
    """Runtime knobs applied on top of the schedule YAML.

    All fields are optional; None means "leave whatever the YAML/defaults say".
    Each field addresses a single axis so they compose freely (e.g. --dry-run
    + --epochs 4 keeps the sample cap but lengthens training).
    """
    dry_run: bool = False
    epochs: Optional[int] = None
    max_samples_per_epoch: Optional[int] = None
    force_eval_every_epoch: bool = False


def _apply_overrides(cfg_dict: Dict[str, Any], ov: StageOverrides) -> None:
    """Mutate the compiled cfg_dict with runtime overrides."""
    training = cfg_dict.setdefault("training", {})
    eval_cfg = cfg_dict.setdefault("eval", {})
    batch_size = training.get("batch_size", 4)

    # Dry-run is the tight preset; individual flags below can widen it.
    if ov.dry_run:
        training["epochs"] = 1
        training["max_batches_per_epoch"] = 20
        training["batch_size"] = min(batch_size, 2)
        batch_size = training["batch_size"]
        # Cap warmup so 20-batch smoke tests still reach real LR (not near-zero).
        if int(training.get("warmup_steps", 0) or 0) > 5:
            training["warmup_steps"] = 5
        eval_cfg["at_epochs"] = [1]
        cfg_dict.setdefault("output", {})["save_interval"] = 1

    if ov.epochs is not None:
        training["epochs"] = int(ov.epochs)

    if ov.max_samples_per_epoch is not None:
        # Convert samples -> batches so downstream (train_one_epoch) stays batch-centric.
        training["max_batches_per_epoch"] = max(1, int(ov.max_samples_per_epoch) // batch_size)

    # Any epoch override can invalidate at_epochs values that exceed the new cap.
    # Guarantee the final epoch is always in the eval set so every stage evaluates
    # at least once at completion. force_eval_every_epoch replaces with a per-epoch sweep.
    final_epoch = int(training.get("epochs", 1))
    at_epochs = eval_cfg.get("at_epochs") or []

    if ov.force_eval_every_epoch:
        eval_cfg["at_epochs"] = list(range(1, final_epoch + 1))
    elif ov.epochs is not None or ov.dry_run:
        reachable = sorted({int(e) for e in at_epochs if 1 <= int(e) <= final_epoch})
        reachable.append(final_epoch)
        eval_cfg["at_epochs"] = sorted(set(reachable))


def compile_stage_cfg(
    schedule: Schedule,
    stage: Stage,
    prev_ckpt: Optional[Path],
    overrides: Optional[StageOverrides] = None,
) -> Config:
    """Produce a fully-resolved Config for one stage.

    Merge order (later wins):
        1. base_config YAML
        2. stage.overrides (from schedule YAML)
        3. scheduler-injected fields (train paths, eval_datasets, output.dir, init_from)
        4. runtime overrides (StageOverrides: dry_run, epochs, max_samples)
    """
    if not schedule.base_config.exists():
        raise FileNotFoundError(f"base_config not found: {schedule.base_config}")

    ov = overrides or StageOverrides()

    cfg_dict = load_yaml(str(schedule.base_config))

    # (2a) common overrides (apply to all stages; stage overrides still win)
    if schedule.common_overrides:
        _deep_merge(cfg_dict, copy.deepcopy(schedule.common_overrides))

    # (2b) stage overrides from YAML
    _deep_merge(cfg_dict, copy.deepcopy(stage.overrides))

    # (3) scheduler-injected fields
    cfg_dict.setdefault("data", {})
    cfg_dict["data"]["train_img_dir"] = stage.train.img_dir
    cfg_dict["data"]["train_ann"] = stage.train.ann_file
    cfg_dict["data"]["eval_datasets"] = _suite_refs_to_eval_datasets(
        stage.eval.suites, schedule.eval_suites, stage.name,
    )
    cfg_dict["data"]["val_img_dir"] = None
    cfg_dict["data"]["val_ann"] = None
    cfg_dict["data"]["coco_o_img_dir"] = None
    cfg_dict["data"]["coco_o_ann"] = None

    cfg_dict.setdefault("output", {})
    cfg_dict["output"]["dir"] = str(schedule.output_root / stage.name)

    cfg_dict.setdefault("training", {})
    cfg_dict["training"]["init_from"] = _resolve_init_from(stage.init_from, prev_ckpt, stage.name)
    cfg_dict["training"]["resume"] = None

    cfg_dict.setdefault("eval", {})
    if stage.eval.at_epochs:
        cfg_dict["eval"]["at_epochs"] = list(stage.eval.at_epochs)

    # (4) runtime overrides
    _apply_overrides(cfg_dict, ov)

    return Config(cfg_dict)


# --------------------------------------------------------------------------- #
# Stage execution
# --------------------------------------------------------------------------- #

def _run_stage(cfg: Config, stage_name: str) -> Dict[str, Any]:
    """Invoke train.run_training in-process. Imported here to avoid cycle."""
    from train import run_training  # local import keeps core/ free of train dep at import time
    _logger.info("=" * 70)
    _logger.info(f"STAGE: {stage_name}")
    _logger.info("=" * 70)
    return run_training(cfg)


def run_schedule(
    schedule_path: Path,
    *,
    only: Optional[str] = None,
    start_from: Optional[str] = None,
    overrides: Optional[StageOverrides] = None,
) -> List[StageResult]:
    """Execute a schedule. Returns a list of StageResult in execution order."""
    schedule = load_schedule(schedule_path)
    schedule.output_root.mkdir(parents=True, exist_ok=True)
    ov = overrides or StageOverrides()

    if schedule.seed is not None:
        _set_global_seed(schedule.seed)

    # Filter stages
    selected = _select_stages(schedule.stages, only=only, start_from=start_from)
    _logger.info(
        f"Schedule: {schedule.name}  |  stages to run: {[s.name for s in selected]}  |  "
        f"dry_run={ov.dry_run}  epochs={ov.epochs}  max_samples={ov.max_samples_per_epoch}"
    )

    results: List[StageResult] = []
    prev_ckpt: Optional[Path] = _find_resumable_prev_ckpt(schedule, selected) if start_from else None

    for stage in selected:
        if not stage.enabled:
            _logger.info(f"Stage '{stage.name}' disabled, skipping.")
            continue

        cfg = compile_stage_cfg(schedule, stage, prev_ckpt=prev_ckpt, overrides=ov)

        # Persist the compiled cfg alongside outputs for reproducibility.
        stage_dir = Path(cfg.output.dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
        )

        summary = _run_stage(cfg, stage.name)
        result = StageResult(
            name=stage.name,
            status=summary.get("status", "unknown"),
            best_ap=float(summary.get("best_ap", -1.0)),
            final_ckpt=summary.get("final_ckpt"),
            output_dir=str(stage_dir),
            summary_path=str(stage_dir / "training_summary.json"),
        )
        results.append(result)

        if result.status != "ok":
            _logger.error(f"Stage '{stage.name}' ended with status={result.status}. Aborting schedule.")
            break

        prev_ckpt = Path(result.final_ckpt) if result.final_ckpt else None

    _write_rollup(schedule, results)
    return results


def _select_stages(
    stages: List[Stage],
    *,
    only: Optional[str],
    start_from: Optional[str],
) -> List[Stage]:
    if only:
        matches = [s for s in stages if s.name == only]
        if not matches:
            raise ValueError(f"--only '{only}' not found. Available: {[s.name for s in stages]}")
        return matches
    if start_from:
        names = [s.name for s in stages]
        if start_from not in names:
            raise ValueError(f"--start-from '{start_from}' not found. Available: {names}")
        idx = names.index(start_from)
        return stages[idx:]
    return list(stages)


def _find_resumable_prev_ckpt(schedule: Schedule, selected: List[Stage]) -> Optional[Path]:
    """When --start-from is used, look for the previous stage's last_epoch.pth on disk."""
    if not selected:
        return None
    names = [s.name for s in schedule.stages]
    first_name = selected[0].name
    idx = names.index(first_name)
    if idx == 0:
        return None
    prev_stage = schedule.stages[idx - 1]
    candidate = schedule.output_root / prev_stage.name / "last_epoch.pth"
    return candidate if candidate.exists() else None


def _write_rollup(schedule: Schedule, results: List[StageResult]) -> None:
    path = schedule.output_root / "schedule_results.json"
    payload = {
        "schedule_name": schedule.name,
        "output_root": str(schedule.output_root),
        "stages": [
            {
                "name": r.name,
                "status": r.status,
                "best_ap": r.best_ap,
                "final_ckpt": r.final_ckpt,
                "output_dir": r.output_dir,
                "summary_path": r.summary_path,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _logger.info(f"Schedule rollup: {path}")


def _set_global_seed(seed: int) -> None:
    import random
    import numpy as np  # type: ignore[import-not-found]
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

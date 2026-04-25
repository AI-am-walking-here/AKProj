"""Detection training: frozen backbone (ViT or CNN) + DETR head.

Two entry points:

1. CLI (single run):
    python train.py --config configs/default.yaml

2. Library (called by schedule.py):
    from train import run_training
    results = run_training(cfg)  # returns dict with best_ap, final_ckpt, per_epoch_metrics

Config extensions for scheduler use (all optional, backward compatible):

    data.eval_datasets:
        - { name: coco,   img_dir: data/coco/val2017,  ann_file: ... }
        - { name: coco_o, img_dir: data/coco-o/images, ann_file: ... }
        If set, replaces the legacy (val_img_dir, coco_o_img_dir) pair.

    eval.at_epochs: [1, 5, 10, -1]
        If set, overrides eval.interval. -1 = final epoch.

    training.init_from: path/to/head.pth
        Load head weights only (fresh optimizer/epoch). Distinct from training.resume
        which restores full training state.

    training.max_batches_per_epoch: 50
        Cap batches per epoch (smoke-test mode).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from core import (
    DetectionModel,
    DetectionLoss,
    HungarianMatcher,
    CocoFormatDataset,
    collate_fn,
    build_category_mapping,
    evaluate_coco_map,
)
from core.config import load_config, Config
from core.det_model import build_detection_model
from core.telemetry import build_sink
from core.transforms import build_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Eval dataset resolution
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvalDatasetSpec:
    """Declarative description of one evaluation dataset."""
    name: str
    img_dir: str
    ann_file: str


def _resolve_eval_specs(cfg: Config) -> List[EvalDatasetSpec]:
    """Return eval datasets from either the new `data.eval_datasets` list or
    the legacy (val_img_dir/val_ann, coco_o_img_dir/coco_o_ann) pair.
    """
    new = getattr(cfg.data, "eval_datasets", None)
    if new:
        out: List[EvalDatasetSpec] = []
        for item in new:
            if isinstance(item, dict):
                out.append(EvalDatasetSpec(name=item["name"], img_dir=item["img_dir"], ann_file=item["ann_file"]))
            else:  # Config wrapper after nested load
                out.append(EvalDatasetSpec(name=item.name, img_dir=item.img_dir, ann_file=item.ann_file))
        return out

    legacy: List[EvalDatasetSpec] = []
    if getattr(cfg.data, "val_img_dir", None) and getattr(cfg.data, "val_ann", None):
        legacy.append(EvalDatasetSpec("coco", cfg.data.val_img_dir, cfg.data.val_ann))
    if getattr(cfg.data, "coco_o_img_dir", None) and getattr(cfg.data, "coco_o_ann", None):
        legacy.append(EvalDatasetSpec("coco_o", cfg.data.coco_o_img_dir, cfg.data.coco_o_ann))
    return legacy


def _resolve_eval_epochs(cfg: Config, total_epochs: int) -> Optional[set]:
    """Return explicit eval-epoch set from cfg.eval.at_epochs, or None if unset."""
    at = getattr(cfg.eval, "at_epochs", None)
    if not at:
        return None
    resolved = {total_epochs if int(e) == -1 else int(e) for e in at}
    return resolved


def _should_eval(epoch: int, total_epochs: int, cfg: Config, explicit: Optional[set]) -> bool:
    if explicit is not None:
        return epoch in explicit
    return (epoch % cfg.eval.interval == 0) or (epoch == total_epochs)


# --------------------------------------------------------------------------- #
# Weight loading: shape-tolerant head init (for cross-dataset transfer)
# --------------------------------------------------------------------------- #

@dataclass
class _LoadReport:
    loaded: int = 0
    dropped_shape: List[Tuple[str, tuple, tuple]] = None
    missing_in_src: List[str] = None
    extra_in_src: List[str] = None

    def __post_init__(self):
        self.dropped_shape = self.dropped_shape or []
        self.missing_in_src = self.missing_in_src or []
        self.extra_in_src = self.extra_in_src or []


def _load_head_shape_tolerant(head: nn.Module, src_state: Dict[str, torch.Tensor]) -> _LoadReport:
    """Load head weights, dropping any key whose shape mismatches the target.

    Intended for cross-dataset transfer where the classifier output dim differs
    (e.g. Objects365's 365-class head -> COCO's 80-class head). Classifier weights
    are dropped and stay at fresh init; everything else transfers.

    Returns a report so the caller can log/inspect what happened.
    """
    dst_state = head.state_dict()
    to_load: Dict[str, torch.Tensor] = {}
    report = _LoadReport()

    for k, v in src_state.items():
        if k not in dst_state:
            report.extra_in_src.append(k)
            continue
        if dst_state[k].shape != v.shape:
            report.dropped_shape.append((k, tuple(v.shape), tuple(dst_state[k].shape)))
            continue
        to_load[k] = v

    missing, _ = head.load_state_dict(to_load, strict=False)
    dropped_keys = {k for k, _, _ in report.dropped_shape}
    report.missing_in_src = [m for m in missing if m not in dropped_keys]
    report.loaded = len(to_load)
    return report


# --------------------------------------------------------------------------- #
# LR warmup (step-level, composes with epoch-level schedulers)
# --------------------------------------------------------------------------- #

def _apply_warmup(
    optimizer: optim.Optimizer,
    global_step: int,
    warmup_steps: int,
    base_lrs: Sequence[float],
) -> None:
    """Linear warmup from 0 -> base_lr over `warmup_steps` steps.

    Safe to call every step: no-op once `global_step >= warmup_steps`. The main
    per-epoch scheduler reads `base_lrs` captured at its init, so warmup
    mutations don't leak into subsequent epochs.
    """
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return
    factor = (global_step + 1) / warmup_steps
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


# --------------------------------------------------------------------------- #
# Training loop (one epoch)
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: DetectionModel,
    criterion: DetectionLoss,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 0.1,
    log_interval: int = 50,
    scaler: Optional[GradScaler] = None,
    max_batches: Optional[int] = None,
    warmup_steps: int = 0,
    global_step_offset: int = 0,
    base_lrs: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    model.train()
    use_amp = scaler is not None
    total_loss = 0.0
    total_cls = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if warmup_steps > 0 and base_lrs is not None:
            _apply_warmup(optimizer, global_step_offset + batch_idx, warmup_steps, base_lrs)

        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        total_cls += loss_dict["loss_cls"].item()
        total_bbox += loss_dict["loss_bbox"].item()
        total_giou += loss_dict["loss_giou"].item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            _logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(data_loader)}]  "
                f"loss={loss.item():.4f}  cls={loss_dict['loss_cls'].item():.4f}  "
                f"bbox={loss_dict['loss_bbox'].item():.4f}  giou={loss_dict['loss_giou'].item():.4f}"
            )

    n = max(num_batches, 1)
    return {
        "train_loss": total_loss / n,
        "train_cls": total_cls / n,
        "train_bbox": total_bbox / n,
        "train_giou": total_giou / n,
        "num_batches": float(num_batches),
    }


# --------------------------------------------------------------------------- #
# Public API — callable by schedule.py
# --------------------------------------------------------------------------- #

def _resolve_device(requested: str) -> torch.device:
    """Pick the best available device, falling back gracefully.

    If the user asked for CUDA but it isn't compiled in, warn once and use CPU
    rather than crashing mid-schedule. Explicit CPU requests pass through.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        _logger.warning(
            f"Requested device='{requested}' but CUDA is not available "
            f"(torch build: {torch.__version__}). Falling back to CPU."
        )
        return torch.device("cpu")
    return torch.device(requested)


def run_training(cfg: Config) -> Dict[str, Any]:
    """Run one training job end-to-end. Returns summary dict.

    Args:
        cfg: Fully-resolved Config (from load_config or compile_stage_cfg).

    Returns:
        {
            "best_ap":         float,
            "final_ckpt":      str,  # path to last_epoch.pth
            "best_ckpt":       str,  # path to best.pth (if any eval ran)
            "per_epoch":       List[Dict[str, Any]],  # train + eval metrics per epoch
            "final_eval":      Dict[str, Dict],       # {dataset_name: AP metrics} at last epoch
            "status":          "ok" | "nan_loss" | "error",
        }
    """
    device = _resolve_device(cfg.device)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sink = build_sink(cfg)

    # ---- training dataset ----
    if not cfg.data.train_img_dir or not cfg.data.train_ann:
        raise ValueError("data.train_img_dir and data.train_ann are required")

    aug_cfg = cfg.augmentation.to_dict() if hasattr(cfg, "augmentation") else {}
    train_transform = build_transforms(aug_cfg, img_size=cfg.backbone.img_size, training=True)
    eval_transform = build_transforms(img_size=cfg.backbone.img_size, training=False)

    train_dataset = CocoFormatDataset(
        img_dir=cfg.data.train_img_dir,
        ann_file=cfg.data.train_ann,
        img_size=cfg.backbone.img_size,
        transform=train_transform,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty (see CocoFormatDataset error above if raised). "
            f"train_img_dir={cfg.data.train_img_dir!r}, train_ann={cfg.data.train_ann!r}. "
            "Fix paths or download images before training."
        )
    num_classes = cfg.data.num_classes or train_dataset.num_classes
    _logger.info(f"Training classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- eval datasets (generalized) ----
    eval_specs = _resolve_eval_specs(cfg)
    # Build (spec, loader, source_to_target, target_label_to_cat_id) tuples.
    eval_bundles: List[Dict[str, Any]] = []
    for spec in eval_specs:
        s2t, t2cat, unmatched = build_category_mapping(
            source_ann_file=cfg.data.train_ann,
            target_ann_file=spec.ann_file,
        )
        if unmatched:
            _logger.info(f"[{spec.name}] unmatched source categories: {len(unmatched)} "
                         f"(first 5: {unmatched[:5]})")
        ds = CocoFormatDataset(
            img_dir=spec.img_dir,
            ann_file=spec.ann_file,
            img_size=cfg.backbone.img_size,
            transform=eval_transform,
        )
        loader = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        eval_bundles.append({
            "spec": spec,
            "loader": loader,
            "source_to_target": s2t,
            "target_label_to_cat_id": t2cat,
        })
    _logger.info(f"Eval datasets: {[b['spec'].name for b in eval_bundles] or 'none'}")

    # ---- model ----
    # Prior bias init only matters for focal loss; CE uses zero bias.
    head_prior_prob = (
        getattr(cfg.head, "prior_prob", 0.01)
        if cfg.loss.cls_type in ("focal", "ia_bce")
        else None
    )
    model = build_detection_model(
        backbone_type=cfg.backbone.type,
        model_name=cfg.backbone.name,
        pretrained=cfg.backbone.pretrained,
        checkpoint_path=cfg.backbone.checkpoint,
        img_size=cfg.backbone.img_size,
        num_classes=num_classes,
        num_queries=cfg.head.num_queries,
        d_model=cfg.head.d_model,
        nhead=cfg.head.nhead,
        num_decoder_layers=cfg.head.num_decoder_layers,
        dim_feedforward=cfg.head.dim_feedforward,
        dropout=cfg.head.dropout,
        aux_loss=cfg.head.aux_loss,
        prior_prob=head_prior_prob,
    )
    model.to(device)

    trainable = sum(p.numel() for p in model.trainable_parameters())
    total = sum(p.numel() for p in model.parameters())
    _logger.info(
        f"Model: {total:,} total | {total - trainable:,} frozen (backbone) | "
        f"{trainable:,} trainable (head)"
    )

    # ---- loss ----
    matcher = HungarianMatcher(
        cost_class=cfg.matcher.cost_class,
        cost_bbox=cfg.matcher.cost_bbox,
        cost_giou=cfg.matcher.cost_giou,
        cls_type=cfg.loss.cls_type,
    )
    criterion = DetectionLoss(
        num_classes=num_classes,
        matcher=matcher,
        cls_type=cfg.loss.cls_type,
        weight_cls=cfg.loss.weight_cls,
        weight_bbox=cfg.loss.weight_bbox,
        weight_giou=cfg.loss.weight_giou,
        eos_coef=cfg.loss.eos_coef,
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma,
    ).to(device)

    _logger.info(f"Loss: cls_type={cfg.loss.cls_type}, "
                 f"weight_cls={cfg.loss.weight_cls}, weight_bbox={cfg.loss.weight_bbox}, "
                 f"weight_giou={cfg.loss.weight_giou}")

    # ---- optimizer ----
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    if cfg.training.lr_schedule == "cosine":
        # eta_min floor: paper-faithful ratio is 0.1 (2.5e-6 from 2.5e-5); legacy default 0.01.
        lr_min_ratio = float(getattr(cfg.training, "lr_min_ratio", 0.01) or 0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.lr * lr_min_ratio,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.lr_drop, gamma=0.1,
        )

    # ---- AMP ----
    use_amp = getattr(cfg.training, "amp", False) and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    _logger.info(f"AMP: {'enabled' if use_amp else 'disabled'}")

    # ---- init_from (head-only load, fresh optimizer) ----
    init_from = getattr(cfg.training, "init_from", None)
    resume_path = getattr(cfg.training, "resume", None)

    start_epoch = 1
    best_ap = -1.0

    if init_from:
        _logger.info(f"Init head weights from: {init_from}")
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        report = _load_head_shape_tolerant(model.head, ckpt["head_state_dict"])
        _logger.info(f"  loaded {report.loaded} tensors from source")
        for key, src_shape, dst_shape in report.dropped_shape:
            _logger.warning(
                f"  [shape-mismatch] '{key}' src={src_shape} vs dst={dst_shape} "
                f"-> DROPPED (fresh init kept). Common when label-space changes."
            )
        if report.missing_in_src:
            _logger.info(
                f"  {len(report.missing_in_src)} dst keys fresh-initialized "
                f"(not in source): e.g. {report.missing_in_src[:3]}"
            )
        if report.extra_in_src:
            _logger.info(f"  {len(report.extra_in_src)} source keys ignored (not in dst)")

    if resume_path:
        _logger.info(f"Resume full training state from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.head.load_state_dict(ckpt["head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ap = ckpt.get("best_ap", -1.0)
        _logger.info(f"  Resumed at epoch {start_epoch}, best_ap={best_ap:.4f}")

    # ---- schedule-specific knobs ----
    max_batches_per_epoch = getattr(cfg.training, "max_batches_per_epoch", None)
    if max_batches_per_epoch:
        _logger.info(f"Training capped at {max_batches_per_epoch} batches/epoch (max_batches_per_epoch)")
    eval_epochs = _resolve_eval_epochs(cfg, cfg.training.epochs)
    if eval_epochs is not None:
        _logger.info(f"Eval at explicit epochs: {sorted(eval_epochs)}")

    # ---- warmup state (skipped on resume: we're past warmup by definition) ----
    warmup_steps = int(getattr(cfg.training, "warmup_steps", 0) or 0)
    if resume_path and warmup_steps > 0:
        _logger.info("Resume path active; skipping LR warmup.")
        warmup_steps = 0
    base_lrs: Optional[List[float]] = (
        [pg["lr"] for pg in optimizer.param_groups] if warmup_steps > 0 else None
    )
    if warmup_steps > 0:
        _logger.info(f"Linear LR warmup: {warmup_steps} steps -> target LR {cfg.training.lr:.2e}")
    global_step = 0

    # ---- training loop ----
    per_epoch: List[Dict[str, Any]] = []
    final_eval: Dict[str, Dict[str, float]] = {}
    status = "ok"
    final_ckpt_path = output_dir / "last_epoch.pth"
    # If the epoch range is empty or we error before the first iteration,
    # checkpoint saving must not reference an unbound `epoch`.
    epoch = max(0, start_epoch - 1)

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            t0 = time.time()
            train_metrics = train_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_grad_norm=cfg.training.max_grad_norm,
                log_interval=cfg.output.log_interval,
                scaler=scaler,
                max_batches=max_batches_per_epoch,
                warmup_steps=warmup_steps,
                global_step_offset=global_step,
                base_lrs=base_lrs,
            )
            global_step += int(train_metrics["num_batches"])
            scheduler.step()
            elapsed = time.time() - t0

            if not (train_metrics["train_loss"] == train_metrics["train_loss"]):  # NaN check
                _logger.error(f"Epoch {epoch}: train_loss is NaN, aborting.")
                status = "nan_loss"
                break

            _logger.info(
                f"Epoch {epoch}/{cfg.training.epochs}  ({elapsed:.1f}s)  "
                f"train_loss={train_metrics['train_loss']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            sink.log_metrics(train_metrics, step=epoch, namespace="train")
            sink.log_metrics({"lr": float(scheduler.get_last_lr()[0])}, step=epoch, namespace="opt")

            epoch_rec: Dict[str, Any] = {"epoch": epoch, "elapsed_s": elapsed, **train_metrics}

            # ---- eval ----
            if _should_eval(epoch, cfg.training.epochs, cfg, eval_epochs):
                for b in eval_bundles:
                    spec: EvalDatasetSpec = b["spec"]
                    _logger.info(f"Evaluating on {spec.name} ...")
                    m = evaluate_coco_map(
                        model=model,
                        data_loader=b["loader"],
                        ann_file=spec.ann_file,
                        target_label_to_cat_id=b["target_label_to_cat_id"],
                        source_to_target_label=b["source_to_target"],
                        device=device,
                        score_threshold=getattr(
                            cfg.eval, "score_threshold", 0.0,
                        ),
                        max_detections=getattr(cfg.eval, "max_detections", 100),
                        cls_type=cfg.loss.cls_type,
                    )
                    _logger.info(
                        f"  {spec.name}  AP={m['AP']:.4f}  AP50={m['AP50']:.4f}  AP75={m['AP75']:.4f}"
                    )
                    sink.log_metrics(m, step=epoch, namespace=spec.name)
                    epoch_rec[f"{spec.name}_AP"] = m["AP"]
                    epoch_rec[f"{spec.name}_AP50"] = m["AP50"]
                    epoch_rec[f"{spec.name}_AP75"] = m["AP75"]
                    final_eval[spec.name] = m

                    # Best checkpoint tracked on FIRST eval dataset (usually coco).
                    if b is eval_bundles[0] and m["AP"] > best_ap:
                        best_ap = m["AP"]
                        torch.save(_build_ckpt(epoch, model, optimizer, scheduler, scaler, best_ap),
                                   output_dir / "best.pth")
                        _logger.info(f"  -> New best AP={best_ap:.4f}, saved best.pth")

            per_epoch.append(epoch_rec)

            # periodic save
            if epoch % cfg.output.save_interval == 0:
                torch.save(_build_ckpt(epoch, model, optimizer, scheduler, scaler, best_ap),
                           output_dir / f"checkpoint_epoch_{epoch}.pth")
    except Exception as e:
        _logger.exception(f"Training failed: {e}")
        status = "error"

    # always write last_epoch.pth (scheduler needs it for @prev chaining)
    torch.save(_build_ckpt(epoch, model, optimizer, scheduler, scaler, best_ap),
               final_ckpt_path)

    summary = {
        "status": status,
        "best_ap": best_ap,
        "final_ckpt": str(final_ckpt_path),
        "best_ckpt": str(output_dir / "best.pth") if (output_dir / "best.pth").exists() else None,
        "per_epoch": per_epoch,
        "final_eval": final_eval,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _logger.info(f"Training complete. Best AP={best_ap:.4f}. Summary: {output_dir/'training_summary.json'}")
    sink.finish()
    return summary


def _build_ckpt(epoch, model, optimizer, scheduler, scaler, best_ap):
    ckpt = {
        "epoch": epoch,
        "head_state_dict": model.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_ap": best_ap,
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    return ckpt


# --------------------------------------------------------------------------- #
# CLI shim
# --------------------------------------------------------------------------- #

def main():
    cfg = load_config()
    run_training(cfg)


if __name__ == "__main__":
    main()

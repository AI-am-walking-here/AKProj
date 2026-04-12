"""Detection training: frozen ViT backbone + DETR head.

Train on Objects365, evaluate COCO mAP on COCO and COCO-O.

Usage:
    python detection_train.py --config configs/default.yaml

    CLI flags override YAML values:
    python detection_train.py --config configs/default.yaml --lr 1e-4 --batch-size 8

Requires: pip install timm (see requirements.txt).
"""
import logging
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from detection import (
    DetectionModel,
    DetectionLoss,
    HungarianMatcher,
    CocoFormatDataset,
    collate_fn,
    build_category_mapping,
    evaluate_coco_map,
)
from detection.config import load_config
from detection.det_model import build_detection_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Training loop
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
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
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
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    cfg = load_config()
    device = torch.device(cfg.device)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- training dataset (Objects365) ----
    if not cfg.data.train_img_dir or not cfg.data.train_ann:
        raise ValueError("--train-img-dir and --train-ann are required (or set in YAML)")

    train_dataset = CocoFormatDataset(
        img_dir=cfg.data.train_img_dir,
        ann_file=cfg.data.train_ann,
        img_size=cfg.backbone.img_size,
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

    # ---- class mapping + eval loaders ----
    has_coco = cfg.data.val_img_dir and cfg.data.val_ann
    has_coco_o = cfg.data.coco_o_img_dir and cfg.data.coco_o_ann

    source_to_target = None
    target_label_to_cat_id = None
    coco_loader = None
    coco_o_loader = None

    if has_coco:
        source_to_target, target_label_to_cat_id, unmatched = build_category_mapping(
            source_ann_file=cfg.data.train_ann,
            target_ann_file=cfg.data.val_ann,
        )
        if unmatched:
            _logger.info(f"Unmatched source categories ({len(unmatched)}): {unmatched[:20]}...")

        coco_dataset = CocoFormatDataset(
            img_dir=cfg.data.val_img_dir,
            ann_file=cfg.data.val_ann,
            img_size=cfg.backbone.img_size,
        )
        coco_loader = DataLoader(
            coco_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    if has_coco_o:
        if target_label_to_cat_id is None:
            _, target_label_to_cat_id, _ = build_category_mapping(
                source_ann_file=cfg.data.train_ann,
                target_ann_file=cfg.data.coco_o_ann,
            )
        if source_to_target is None:
            source_to_target, _, _ = build_category_mapping(
                source_ann_file=cfg.data.train_ann,
                target_ann_file=cfg.data.coco_o_ann,
            )

        coco_o_dataset = CocoFormatDataset(
            img_dir=cfg.data.coco_o_img_dir,
            ann_file=cfg.data.coco_o_ann,
            img_size=cfg.backbone.img_size,
        )
        coco_o_loader = DataLoader(
            coco_o_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # ---- model ----
    model = build_detection_model(
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

    # ---- optimizer (head only) ----
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    if cfg.training.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 0.01,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.lr_drop, gamma=0.1,
        )

    # ---- training loop ----
    best_ap = -1.0

    for epoch in range(1, cfg.training.epochs + 1):
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
        )
        scheduler.step()
        elapsed = time.time() - t0

        _logger.info(
            f"Epoch {epoch}/{cfg.training.epochs}  ({elapsed:.1f}s)  "
            f"train_loss={train_metrics['train_loss']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # ---- COCO mAP evaluation ----
        run_eval = (epoch % cfg.eval.interval == 0) or (epoch == cfg.training.epochs)

        if run_eval and coco_loader is not None:
            _logger.info("Evaluating on COCO val ...")
            coco_metrics = evaluate_coco_map(
                model=model,
                data_loader=coco_loader,
                ann_file=cfg.data.val_ann,
                target_label_to_cat_id=target_label_to_cat_id,
                source_to_target_label=source_to_target,
                device=device,
            )
            _logger.info(
                f"  COCO  AP={coco_metrics['AP']:.4f}  "
                f"AP50={coco_metrics['AP50']:.4f}  "
                f"AP75={coco_metrics['AP75']:.4f}"
            )

            if coco_metrics["AP"] > best_ap:
                best_ap = coco_metrics["AP"]
                torch.save(
                    {
                        "epoch": epoch,
                        "head_state_dict": model.head.state_dict(),
                        "ap": best_ap,
                    },
                    output_dir / "best.pth",
                )
                _logger.info(f"  -> New best AP={best_ap:.4f}, saved best.pth")

        if run_eval and coco_o_loader is not None:
            _logger.info("Evaluating on COCO-O ...")
            coco_o_metrics = evaluate_coco_map(
                model=model,
                data_loader=coco_o_loader,
                ann_file=cfg.data.coco_o_ann,
                target_label_to_cat_id=target_label_to_cat_id,
                source_to_target_label=source_to_target,
                device=device,
            )
            _logger.info(
                f"  COCO-O  AP={coco_o_metrics['AP']:.4f}  "
                f"AP50={coco_o_metrics['AP50']:.4f}  "
                f"AP75={coco_o_metrics['AP75']:.4f}"
            )

        # ---- periodic checkpoint ----
        if epoch % cfg.output.save_interval == 0:
            torch.save(
                {"epoch": epoch, "head_state_dict": model.head.state_dict()},
                output_dir / f"checkpoint_epoch_{epoch}.pth",
            )

    _logger.info(f"Training complete.  Best COCO AP={best_ap:.4f}")


if __name__ == "__main__":
    main()

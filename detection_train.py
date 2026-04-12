"""Detection training: frozen ViT backbone + DETR head.

Train on Objects365, evaluate COCO mAP on COCO and COCO-O.

Usage:
    python detection_train.py \
        --train-img-dir /data/objects365/train \
        --train-ann     /data/objects365/annotations/train.json \
        --val-img-dir   /data/coco/val2017 \
        --val-ann       /data/coco/annotations/instances_val2017.json \
        --checkpoint    checkpoints/labelmix/model_best.pth.tar \
        --epochs 50 --lr 1e-4 --batch-size 4

Optionally add COCO-O evaluation:
        --coco-o-img-dir /data/coco-o/images \
        --coco-o-ann     /data/coco-o/annotations/coco_o.json

Requires: pytorch-image-models/ on sys.path (handled automatically).
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent
_timm_root = _project_root / "pytorch-image-models"
if str(_timm_root) not in sys.path:
    sys.path.insert(0, str(_timm_root))

from detection import (
    DetectionModel,
    DetectionLoss,
    HungarianMatcher,
    CocoFormatDataset,
    collate_fn,
    build_category_mapping,
    evaluate_coco_map,
)
from detection.det_model import build_detection_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Training / loss-eval loops
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
    total_ce = 0.0
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
        total_ce += loss_dict["loss_ce"].item()
        total_bbox += loss_dict["loss_bbox"].item()
        total_giou += loss_dict["loss_giou"].item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            _logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(data_loader)}]  "
                f"loss={loss.item():.4f}  ce={loss_dict['loss_ce'].item():.4f}  "
                f"bbox={loss_dict['loss_bbox'].item():.4f}  giou={loss_dict['loss_giou'].item():.4f}"
            )

    n = max(num_batches, 1)
    return {
        "train_loss": total_loss / n,
        "train_ce": total_ce / n,
        "train_bbox": total_bbox / n,
        "train_giou": total_giou / n,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="DETR detection training with frozen ViT backbone")

    # --- training data (e.g. Objects365) ---
    p.add_argument("--train-img-dir", type=str, required=True,
                   help="Image directory for training (e.g. Objects365 train images)")
    p.add_argument("--train-ann", type=str, required=True,
                   help="COCO-format annotation JSON for training")

    # --- eval data: COCO ---
    p.add_argument("--val-img-dir", type=str, default=None,
                   help="Image directory for COCO val evaluation")
    p.add_argument("--val-ann", type=str, default=None,
                   help="COCO-format annotation JSON for COCO val")

    # --- eval data: COCO-O ---
    p.add_argument("--coco-o-img-dir", type=str, default=None,
                   help="Image directory for COCO-O evaluation")
    p.add_argument("--coco-o-ann", type=str, default=None,
                   help="COCO-format annotation JSON for COCO-O")

    p.add_argument("--img-size", type=int, default=256)

    # --- backbone ---
    p.add_argument("--backbone", type=str, default="vit_base_patch16_rope_reg1_gap_256")
    p.add_argument("--pretrained", action="store_true",
                   help="Load timm pretrained backbone weights")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Local backbone checkpoint path")

    # --- DETR head ---
    p.add_argument("--num-classes", type=int, default=None,
                   help="Override class count (auto-detected from training annotations)")
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-decoder-layers", type=int, default=6)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-aux-loss", action="store_true")

    # --- loss / matcher weights ---
    p.add_argument("--cost-class", type=float, default=1.0)
    p.add_argument("--cost-bbox", type=float, default=5.0)
    p.add_argument("--cost-giou", type=float, default=2.0)
    p.add_argument("--weight-ce", type=float, default=1.0)
    p.add_argument("--weight-bbox", type=float, default=5.0)
    p.add_argument("--weight-giou", type=float, default=2.0)
    p.add_argument("--eos-coef", type=float, default=0.1)

    # --- training ---
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-drop", type=int, default=40)
    p.add_argument("--max-grad-norm", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)

    # --- eval schedule ---
    p.add_argument("--eval-interval", type=int, default=1,
                   help="Run COCO mAP evaluation every N epochs")

    # --- output ---
    p.add_argument("--output-dir", type=str, default="output/detection")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=5)

    # --- device ---
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- training dataset (Objects365) ----
    train_dataset = CocoFormatDataset(
        img_dir=args.train_img_dir,
        ann_file=args.train_ann,
        img_size=args.img_size,
    )
    num_classes = args.num_classes or train_dataset.num_classes
    _logger.info(f"Training classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- class mapping + eval loaders ----
    has_coco = args.val_img_dir and args.val_ann
    has_coco_o = args.coco_o_img_dir and args.coco_o_ann

    source_to_target = None
    target_label_to_cat_id = None
    coco_loader = None
    coco_o_loader = None

    if has_coco:
        source_to_target, target_label_to_cat_id, unmatched = build_category_mapping(
            source_ann_file=args.train_ann,
            target_ann_file=args.val_ann,
        )
        if unmatched:
            _logger.info(f"Unmatched source categories ({len(unmatched)}): {unmatched[:20]}...")

        coco_dataset = CocoFormatDataset(
            img_dir=args.val_img_dir,
            ann_file=args.val_ann,
            img_size=args.img_size,
        )
        coco_loader = DataLoader(
            coco_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    if has_coco_o:
        if target_label_to_cat_id is None:
            _, target_label_to_cat_id, _ = build_category_mapping(
                source_ann_file=args.train_ann,
                target_ann_file=args.coco_o_ann,
            )
        if source_to_target is None:
            source_to_target, _, _ = build_category_mapping(
                source_ann_file=args.train_ann,
                target_ann_file=args.coco_o_ann,
            )

        coco_o_dataset = CocoFormatDataset(
            img_dir=args.coco_o_img_dir,
            ann_file=args.coco_o_ann,
            img_size=args.img_size,
        )
        coco_o_loader = DataLoader(
            coco_o_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # ---- model ----
    model = build_detection_model(
        model_name=args.backbone,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        num_classes=num_classes,
        num_queries=args.num_queries,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        aux_loss=not args.no_aux_loss,
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
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
    )
    criterion = DetectionLoss(
        num_classes=num_classes,
        matcher=matcher,
        weight_ce=args.weight_ce,
        weight_bbox=args.weight_bbox,
        weight_giou=args.weight_giou,
        eos_coef=args.eos_coef,
    ).to(device)

    # ---- optimizer (head only) ----
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    # ---- training loop ----
    best_ap = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
        )
        scheduler.step()
        elapsed = time.time() - t0

        _logger.info(
            f"Epoch {epoch}/{args.epochs}  ({elapsed:.1f}s)  "
            f"train_loss={train_metrics['train_loss']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # ---- COCO mAP evaluation ----
        run_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs)

        if run_eval and coco_loader is not None:
            _logger.info("Evaluating on COCO val ...")
            coco_metrics = evaluate_coco_map(
                model=model,
                data_loader=coco_loader,
                ann_file=args.val_ann,
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
                ann_file=args.coco_o_ann,
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
        if epoch % args.save_interval == 0:
            torch.save(
                {"epoch": epoch, "head_state_dict": model.head.state_dict()},
                output_dir / f"checkpoint_epoch_{epoch}.pth",
            )

    _logger.info(f"Training complete.  Best COCO AP={best_ap:.4f}")


if __name__ == "__main__":
    main()

"""Detection training script for frozen ViT backbone + DETR head.

Usage:
    python detection_train.py \
        --coco-path /path/to/coco \
        --backbone vit_base_patch16_rope_reg1_gap_256 \
        --checkpoint checkpoints/labelmix/model.pth \
        --epochs 50 \
        --lr 1e-4 \
        --batch-size 4

Requires: pytorch-image-models/ on sys.path (handled automatically).
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_project_root = Path(__file__).resolve().parent
_timm_root = _project_root / "pytorch-image-models"
if str(_timm_root) not in sys.path:
    sys.path.insert(0, str(_timm_root))

from detection import DetectionModel, DetectionLoss, HungarianMatcher
from detection.det_model import build_detection_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COCO Dataset
# ---------------------------------------------------------------------------
class CocoDetectionDataset(Dataset):
    """Minimal COCO-format dataset for detection training.

    Expects standard COCO directory layout:
        coco_path/
            train2017/
            val2017/
            annotations/
                instances_train2017.json
                instances_val2017.json

    Args:
        root: Path to image directory (e.g. coco_path/train2017).
        ann_file: Path to annotation JSON.
        img_size: Resize images to (img_size, img_size).
        max_detections: Cap on number of GT boxes per image.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        img_size: int = 256,
        max_detections: int = 100,
    ):
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.max_detections = max_detections

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.cat_to_label = {
            cat["id"]: i for i, cat in enumerate(coco["categories"])
        }
        self.num_classes = len(self.cat_to_label)

        self.img_anns: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            img_id = ann["image_id"]
            self.img_anns.setdefault(img_id, []).append(ann)

        self.img_ids = [
            img_id for img_id in self.images
            if img_id in self.img_anns and len(self.img_anns[img_id]) > 0
        ]

        _logger.info(
            f"CocoDetectionDataset: {len(self.img_ids)} images, "
            f"{self.num_classes} classes from {ann_file}"
        )

        try:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        except ImportError:
            raise ImportError("torchvision is required for image transforms")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from PIL import Image

        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.root / img_info["file_name"]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img_tensor = self.transform(img)

        anns = self.img_anns[img_id][:self.max_detections]

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat_to_label[ann["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            "image_id": torch.tensor([img_id]),
        }

        return img_tensor, target


def collate_fn(batch):
    """Custom collate that keeps per-image target dicts as a list."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
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

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_ce": total_ce / max(num_batches, 1),
        "train_bbox": total_bbox / max(num_batches, 1),
        "train_giou": total_giou / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: DetectionModel,
    criterion: DetectionLoss,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)

        total_loss += loss_dict["loss"].item()
        total_ce += loss_dict["loss_ce"].item()
        total_bbox += loss_dict["loss_bbox"].item()
        total_giou += loss_dict["loss_giou"].item()
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "val_loss": total_loss / n,
        "val_ce": total_ce / n,
        "val_bbox": total_bbox / n,
        "val_giou": total_giou / n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DETR detection training with frozen ViT backbone")

    # Data
    p.add_argument("--coco-path", type=str, required=True, help="Root of COCO dataset")
    p.add_argument("--train-split", type=str, default="train2017")
    p.add_argument("--val-split", type=str, default="val2017")
    p.add_argument("--img-size", type=int, default=256)

    # Backbone
    p.add_argument("--backbone", type=str, default="vit_base_patch16_rope_reg1_gap_256")
    p.add_argument("--pretrained", action="store_true", help="Load timm pretrained backbone weights")
    p.add_argument("--checkpoint", type=str, default=None, help="Local backbone checkpoint path")

    # DETR head
    p.add_argument("--num-classes", type=int, default=None, help="Auto-detected from COCO if None")
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-decoder-layers", type=int, default=6)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-aux-loss", action="store_true")

    # Loss weights
    p.add_argument("--cost-class", type=float, default=1.0, help="Matcher class cost")
    p.add_argument("--cost-bbox", type=float, default=5.0, help="Matcher L1 cost")
    p.add_argument("--cost-giou", type=float, default=2.0, help="Matcher GIoU cost")
    p.add_argument("--weight-ce", type=float, default=1.0)
    p.add_argument("--weight-bbox", type=float, default=5.0)
    p.add_argument("--weight-giou", type=float, default=2.0)
    p.add_argument("--eos-coef", type=float, default=0.1, help="No-object class weight")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-drop", type=int, default=40, help="Epoch to drop LR by 10x")
    p.add_argument("--max-grad-norm", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)

    # Output
    p.add_argument("--output-dir", type=str, default="output/detection")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=5)

    # Device
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    coco_root = Path(args.coco_path)
    train_dataset = CocoDetectionDataset(
        root=str(coco_root / args.train_split),
        ann_file=str(coco_root / "annotations" / f"instances_{args.train_split}.json"),
        img_size=args.img_size,
    )
    val_dataset = CocoDetectionDataset(
        root=str(coco_root / args.val_split),
        ann_file=str(coco_root / "annotations" / f"instances_{args.val_split}.json"),
        img_size=args.img_size,
    )

    num_classes = args.num_classes or train_dataset.num_classes
    _logger.info(f"Detected {num_classes} classes from dataset")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- Model ---
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

    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    _logger.info(
        f"Model built: {total_params:,} total params | "
        f"{frozen_params:,} frozen (backbone) | "
        f"{trainable_params:,} trainable (head)"
    )

    # --- Loss ---
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

    # --- Optimizer (head only) ---
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    # --- Training ---
    best_val_loss = float("inf")

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

        val_metrics = evaluate(
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            device=device,
        )

        scheduler.step()

        elapsed = time.time() - t0
        _logger.info(
            f"Epoch {epoch}/{args.epochs}  ({elapsed:.1f}s)  "
            f"train_loss={train_metrics['train_loss']:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {"epoch": epoch, "head_state_dict": model.head.state_dict(), "val_loss": best_val_loss},
                output_dir / "best.pth",
            )
            _logger.info(f"  -> New best val_loss={best_val_loss:.4f}, saved best.pth")

        if epoch % args.save_interval == 0:
            torch.save(
                {"epoch": epoch, "head_state_dict": model.head.state_dict()},
                output_dir / f"checkpoint_epoch_{epoch}.pth",
            )

    _logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()

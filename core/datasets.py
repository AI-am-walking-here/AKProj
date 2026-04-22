"""COCO-format dataset for detection training and evaluation.

Works with any dataset using COCO-style annotation JSON:
Objects365, COCO, COCO-O, LVIS, etc.
"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)


class CocoFormatDataset(Dataset):
    """Dataset for any COCO-format annotation file.

    Loads images and bounding-box annotations from a COCO-style JSON.
    Boxes are returned as normalised (cx, cy, w, h) in [0, 1].

    Args:
        img_dir: Directory containing images.
        ann_file: Path to COCO-format annotation JSON.
        img_size: Resize images to (img_size, img_size).
        max_detections: Cap on GT boxes per image.
        category_mapping: Optional dict {original_cat_id: label_index}.
            If ``None``, categories are mapped to 0 .. K-1
            in the order they appear in the annotation file.
        transform: Optional ``(PIL.Image, target) -> (Tensor, target)``
            pipeline. If ``None``, a default resize + normalize is used.
            Use ``detection.transforms.build_transforms`` to construct one.
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: int = 256,
        max_detections: int = 100,
        category_mapping: Optional[Dict[int, int]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.max_detections = max_detections

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = {cat["id"]: cat for cat in coco["categories"]}

        if category_mapping is not None:
            self.cat_to_label = category_mapping
        else:
            self.cat_to_label = {
                cat["id"]: i for i, cat in enumerate(coco["categories"])
            }

        self.num_classes = (max(self.cat_to_label.values()) + 1) if self.cat_to_label else 0

        self.img_anns: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            if ann["category_id"] not in self.cat_to_label:
                continue
            self.img_anns.setdefault(ann["image_id"], []).append(ann)

        self.img_ids = [
            img_id for img_id in self.images
            if img_id in self.img_anns and len(self.img_anns[img_id]) > 0
        ]

        _logger.info(
            f"CocoFormatDataset: {len(self.img_ids)} images, "
            f"{self.num_classes} classes from {ann_file}"
        )

        if transform is not None:
            self.transform = transform
        else:
            from .transforms import build_transforms
            self.transform = build_transforms(img_size=img_size, training=False)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        from PIL import Image

        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info["file_name"]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        anns = self.img_anns[img_id][: self.max_detections]

        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([(x + w / 2) / orig_w, (y + h / 2) / orig_h, w / orig_w, h / orig_h])
            labels.append(self.cat_to_label[ann["category_id"]])

        target: Dict[str, Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([orig_w, orig_h]),
        }

        img_tensor, target = self.transform(img, target)
        return img_tensor, target


def collate_fn(batch):
    """Stack images, keep per-image target dicts as a list."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

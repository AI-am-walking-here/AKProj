"""COCO-format dataset for detection training and evaluation.

Works with any dataset using COCO-style annotation JSON:
Objects365, COCO, COCO-O, LVIS, etc.
"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)

# Corrupt-image handling knobs. Tuned for "fail loudly at first, then power through":
#   - Warn on each distinct bad path the first time we see it (up to a cap).
#   - Skip forward up to BAD_IMAGE_MAX_SKIP indices before giving up.
BAD_IMAGE_WARN_CAP = 20
BAD_IMAGE_MAX_SKIP = 50


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

        candidate_ids = [
            img_id for img_id in self.images
            if img_id in self.img_anns and len(self.img_anns[img_id]) > 0
        ]
        n_candidates = len(candidate_ids)

        # Filter out missing / zero-byte files up front. Catches truncated downloads
        # cheaply (no decode). Truly corrupt-but-nonzero files are handled lazily in
        # __getitem__ so we don't pay full-decode cost here.
        self.img_ids, skipped = self._filter_readable(candidate_ids)
        if skipped:
            _logger.warning(
                f"  skipped {len(skipped)} missing/zero-byte images "
                f"(first 3: {[s.name for s in skipped[:3]]})"
            )

        if not self.img_ids:
            raise ValueError(
                "CocoFormatDataset: zero usable images after filtering. "
                f"{n_candidates} annotation entries pointed to files under {self.img_dir}, "
                f"but none were readable (missing or zero-byte). "
                f"Ann file: {ann_file}. "
                "Install images so each COCO `file_name` exists under `img_dir` "
                "(e.g. unzip train2017 into data/coco/train2017)."
            )

        # Runtime bad-image bookkeeping (populated in __getitem__ as errors surface).
        self._bad_indices: Set[int] = set()
        self._bad_paths_warned: Set[str] = set()

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

    # -- fsck helpers ------------------------------------------------------- #

    def _filter_readable(self, candidate_ids: List[int]) -> Tuple[List[int], List[Path]]:
        """Drop ids whose file is missing or zero bytes (truncated download)."""
        keep: List[int] = []
        dropped: List[Path] = []
        for img_id in candidate_ids:
            path = self.img_dir / self.images[img_id]["file_name"]
            try:
                st = path.stat()
            except OSError:
                dropped.append(path)
                continue
            if st.st_size == 0:
                dropped.append(path)
                continue
            keep.append(img_id)
        return keep, dropped

    # -- decode + transform (pure, no error handling) ----------------------- #

    def _load_sample(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Load and transform one sample. Raises on any decode failure."""
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

    # -- public access: decode + graceful skip on failure ------------------- #

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        from PIL import UnidentifiedImageError

        start_idx = idx
        for attempt in range(BAD_IMAGE_MAX_SKIP):
            try_idx = (start_idx + attempt) % len(self.img_ids)

            # Skip ids we've already flagged corrupt in this worker.
            if try_idx in self._bad_indices:
                continue

            try:
                return self._load_sample(try_idx)
            except (UnidentifiedImageError, OSError, SyntaxError) as e:
                img_id = self.img_ids[try_idx]
                path = str(self.img_dir / self.images[img_id]["file_name"])
                self._bad_indices.add(try_idx)
                if (path not in self._bad_paths_warned
                        and len(self._bad_paths_warned) < BAD_IMAGE_WARN_CAP):
                    self._bad_paths_warned.add(path)
                    _logger.warning(f"[bad image] {path}: {type(e).__name__}: {e}")
                # fall through to next attempt
        raise RuntimeError(
            f"CocoFormatDataset: could not load any valid image within "
            f"{BAD_IMAGE_MAX_SKIP} indices starting from {start_idx}. "
            f"Dataset is badly corrupted — abort."
        )


def collate_fn(batch):
    """Stack images, keep per-image target dicts as a list."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

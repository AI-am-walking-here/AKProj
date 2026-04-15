"""Detection-aware image + box transforms.

Each transform operates on an ``(image, target)`` pair so that spatial
augmentations (flip, resize, crop) keep bounding boxes consistent.

All box coordinates are normalised ``(cx, cy, w, h)`` in ``[0, 1]``.

Usage:
    pipeline = build_transforms(cfg_dict, img_size=256, training=True)
    image, target = pipeline(image, target)

The ``build_transforms`` factory reads a config dict to toggle each
augmentation on/off, so experiments are a YAML change, not a code change.
"""
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


# --------------------------------------------------------------------------- #
# Primitive transforms (image + target)
# --------------------------------------------------------------------------- #

class Compose:
    """Chain multiple (image, target) transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self, image: Image.Image, target: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ResizeAndNormalize:
    """Resize to fixed square size, convert to tensor, ImageNet-normalize."""

    def __init__(self, img_size: int):
        self.img_size = img_size

    def __call__(
        self, image: Image.Image, target: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        image = F.resize(image, [self.img_size, self.img_size])
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        return image, target


class RandomHorizontalFlip:
    """Flip image and mirror box cx coordinates with probability ``p``."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: Image.Image, target: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        if random.random() >= self.p:
            return image, target

        image = F.hflip(image)
        boxes = target["boxes"]
        if boxes.numel() > 0:
            cx, cy, w, h = boxes.unbind(-1)
            boxes = torch.stack([1.0 - cx, cy, w, h], dim=-1)
            target = {**target, "boxes": boxes}

        return image, target


class RandomResize:
    """Resize the short side to a random value from ``sizes``.

    Maintains aspect ratio, then crops/pads to ``max_size`` if needed.
    Boxes stay in normalised coords so no adjustment is required.
    """

    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(
        self, image: Image.Image, target: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        size = random.choice(self.sizes)
        w, h = image.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        if self.max_size is not None:
            max_scale = self.max_size / max(new_w, new_h)
            if max_scale < 1.0:
                new_w = int(new_w * max_scale)
                new_h = int(new_h * max_scale)

        image = F.resize(image, [new_h, new_w])
        return image, target


class RandomSizeCrop:
    """Crop a random region, discarding boxes whose center falls outside.

    Boxes are in normalised ``(cx, cy, w, h)`` — they get re-normalised
    to the cropped region.
    """

    def __init__(self, min_scale: float = 0.5, max_scale: float = 1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(
        self, image: Image.Image, target: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = image.size
        scale = random.uniform(self.min_scale, self.max_scale)
        crop_w = int(w * scale)
        crop_h = int(h * scale)

        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)

        image = F.crop(image, top, left, crop_h, crop_w)

        boxes = target["boxes"]
        if boxes.numel() == 0:
            return image, target

        cx, cy, bw, bh = boxes.unbind(-1)
        abs_cx = cx * w
        abs_cy = cy * h
        abs_bw = bw * w
        abs_bh = bh * h

        new_cx = (abs_cx - left) / crop_w
        new_cy = (abs_cy - top) / crop_h
        new_bw = abs_bw / crop_w
        new_bh = abs_bh / crop_h

        keep = (new_cx > 0) & (new_cx < 1) & (new_cy > 0) & (new_cy < 1)

        new_cx = new_cx.clamp(0.0, 1.0)
        new_cy = new_cy.clamp(0.0, 1.0)
        new_bw = new_bw.clamp(0.0, 1.0)
        new_bh = new_bh.clamp(0.0, 1.0)

        boxes = torch.stack([new_cx, new_cy, new_bw, new_bh], dim=-1)
        target = {
            **target,
            "boxes": boxes[keep],
            "labels": target["labels"][keep],
        }

        return image, target


class ColorJitter:
    """Random brightness, contrast, saturation, hue perturbation."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ):
        self.jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(
        self, image: Image.Image, target: Dict[str, Any],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        image = self.jitter(image)
        return image, target


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_transforms(
    augmentation_cfg: Optional[Dict[str, Any]] = None,
    img_size: int = 256,
    training: bool = True,
) -> Compose:
    """Build a detection transform pipeline from a config dict.

    Args:
        augmentation_cfg: Dict from ``cfg.data.augmentation``.
            Keys (all optional, defaults shown):
                ``horizontal_flip``: bool (True)
                ``color_jitter``: bool (False)
                ``multiscale``: list[int] | None (None)
                ``random_crop``: bool (False)
                ``crop_min_scale``: float (0.5)
        img_size: Final square output size.
        training: If False, only resize + normalize (no augmentation).

    Returns:
        Compose pipeline that takes ``(PIL.Image, target_dict)``
        and returns ``(Tensor, target_dict)``.
    """
    if augmentation_cfg is None:
        augmentation_cfg = {}

    if not training:
        return Compose([ResizeAndNormalize(img_size)])

    ops: List[Callable] = []

    if augmentation_cfg.get("horizontal_flip", True):
        ops.append(RandomHorizontalFlip(p=0.5))

    if augmentation_cfg.get("color_jitter", False):
        ops.append(ColorJitter())

    multiscale = augmentation_cfg.get("multiscale")
    if multiscale:
        ops.append(RandomResize(sizes=multiscale, max_size=img_size * 2))

    if augmentation_cfg.get("random_crop", False):
        min_scale = augmentation_cfg.get("crop_min_scale", 0.5)
        ops.append(RandomSizeCrop(min_scale=min_scale))

    ops.append(ResizeAndNormalize(img_size))

    return Compose(ops)

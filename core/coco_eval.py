"""COCO-style mAP evaluation with cross-dataset class remapping.

Converts model predictions to COCO result format, optionally remaps
class indices (e.g. Objects365 → COCO), and runs the standard
pycocotools COCO evaluator.
"""
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .metrics.coco_metrics import run_coco_evaluation  # noqa: F401  re-exported for backward compat

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prediction conversion
# --------------------------------------------------------------------------- #

def _build_remap_tensor(
    source_to_target_label: Dict[int, int],
    num_source_classes: int,
    device: torch.device,
) -> Tensor:
    """Return a (num_source_classes,) tensor: src → tgt, or -1 if unmapped."""
    remap = torch.full((num_source_classes,), -1, dtype=torch.long, device=device)
    for src, tgt in source_to_target_label.items():
        remap[src] = tgt
    return remap


def predictions_to_coco_results(
    pred_logits: Tensor,
    pred_boxes: Tensor,
    image_ids: List[int],
    orig_sizes: Tensor,
    label_to_cat_id: Dict[int, int],
    source_to_target_label: Optional[Dict[int, int]] = None,
    score_threshold: float = 0.01,
    max_detections: int = 100,
    cls_type: str = "focal",
) -> List[Dict]:
    """Convert a batch of model outputs to COCO result dicts.

    Args:
        pred_logits: (B, Q, num_classes+1) raw logits.
        pred_boxes: (B, Q, 4) normalised cxcywh.
        image_ids: COCO image ID per batch element.
        orig_sizes: (B, 2) tensor of (orig_w, orig_h).
        label_to_cat_id: target label index → COCO category ID.
        source_to_target_label: Optional source→target label remap.
        score_threshold: Minimum confidence to keep a prediction.
        max_detections: Maximum predictions per image.
        cls_type: ``"focal"`` (sigmoid, independent per-class scores) or
            ``"cross_entropy"`` (softmax over all classes including background).
            Must match the loss used during training.

    Returns:
        List of ``{"image_id", "category_id", "bbox", "score"}`` dicts
        ready for ``pycocotools.coco.COCO.loadRes``.
    """
    B, Q, C_plus_1 = pred_logits.shape
    num_fg = C_plus_1 - 1
    device = pred_logits.device

    if cls_type == "focal":
        # Sigmoid focal loss: each class is an independent binary classifier.
        # The last logit channel is unused background — slice it off before sigmoid.
        probs = torch.sigmoid(pred_logits[:, :, :num_fg])   # (B, Q, num_fg)
    else:
        # Softmax cross-entropy: background is the final class; slice after softmax.
        probs = F.softmax(pred_logits, dim=-1)[:, :, :num_fg]

    scores, labels = probs.max(dim=-1)                      # (B, Q)

    if source_to_target_label is not None:
        remap = _build_remap_tensor(source_to_target_label, num_fg, device)
        remapped = remap[labels]   # (B, Q)  — -1 where unmapped
        valid = remapped >= 0
    else:
        remapped = labels
        valid = torch.ones(B, Q, dtype=torch.bool, device=device)

    results: List[Dict] = []
    for b in range(B):
        keep = valid[b] & (scores[b] > score_threshold)
        b_scores = scores[b][keep]
        b_labels = remapped[b][keep]
        b_boxes = pred_boxes[b][keep]

        if b_scores.numel() > max_detections:
            topk = b_scores.topk(max_detections).indices
            b_scores = b_scores[topk]
            b_labels = b_labels[topk]
            b_boxes = b_boxes[topk]

        ow, oh = orig_sizes[b][0].item(), orig_sizes[b][1].item()
        cx, cy, w, h = b_boxes.unbind(-1)
        abs_x = (cx - w / 2) * ow
        abs_y = (cy - h / 2) * oh
        abs_w = w * ow
        abs_h = h * oh

        for j in range(b_scores.shape[0]):
            lbl = b_labels[j].item()
            if lbl not in label_to_cat_id:
                continue
            results.append({
                "image_id": image_ids[b],
                "category_id": label_to_cat_id[lbl],
                "bbox": [
                    round(abs_x[j].item(), 2),
                    round(abs_y[j].item(), 2),
                    round(abs_w[j].item(), 2),
                    round(abs_h[j].item(), 2),
                ],
                "score": round(b_scores[j].item(), 4),
            })

    return results


# --------------------------------------------------------------------------- #
# Full evaluation loop
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_coco_map(
    model: nn.Module,
    data_loader: DataLoader,
    ann_file: str,
    target_label_to_cat_id: Dict[int, int],
    source_to_target_label: Optional[Dict[int, int]] = None,
    device: torch.device = torch.device("cpu"),
    score_threshold: float = 0.01,
    max_detections: int = 100,
    cls_type: str = "focal",
) -> Dict[str, float]:
    """Run model on *data_loader* and compute COCO mAP against *ann_file*.

    Args:
        model: Detection model returning ``{pred_logits, pred_boxes}``.
        data_loader: Yields ``(images, targets)`` where each target
            contains ``image_id`` and ``orig_size``.
        ann_file: Ground-truth COCO annotation JSON.
        target_label_to_cat_id: target label → COCO category ID.
        source_to_target_label: Optional source→target label remap
            (needed when the model was trained on a different dataset).
        device: Torch device.
        score_threshold: Min score for predictions.
        max_detections: Max predictions per image.
        cls_type: ``"focal"`` or ``"cross_entropy"`` — must match the
            loss used during training. Controls whether sigmoid or softmax
            is applied to raw logits at inference time.

    Returns:
        Dict of COCO metrics (AP, AP50, AP75, …).
    """
    model.eval()
    all_results: List[Dict] = []

    for images, targets in data_loader:
        images = images.to(device)
        outputs = model(images)

        image_ids = [t["image_id"].item() for t in targets]
        orig_sizes = torch.stack([t["orig_size"] for t in targets])

        batch_results = predictions_to_coco_results(
            pred_logits=outputs["pred_logits"],
            pred_boxes=outputs["pred_boxes"],
            image_ids=image_ids,
            orig_sizes=orig_sizes,
            label_to_cat_id=target_label_to_cat_id,
            source_to_target_label=source_to_target_label,
            score_threshold=score_threshold,
            max_detections=max_detections,
            cls_type=cls_type,
        )
        all_results.extend(batch_results)

    return run_coco_evaluation(all_results, ann_file)

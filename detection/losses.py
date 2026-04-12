"""Detection losses: classification (focal) + bbox (L1 + GIoU).

All loss weights are injected via constructor — nothing hardcoded.
Operates on matched prediction-target pairs produced by HungarianMatcher.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """Focal loss for dense classification (Lin et al., 2017).

    Operates on raw logits (applies sigmoid internally).

    Args:
        inputs: (N, C) raw logits.
        targets: (N, C) one-hot or soft targets.
        alpha: Weighting factor for positive class.
        gamma: Focusing parameter.
        reduction: 'mean', 'sum', or 'none'.
    """
    p = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class DetectionLoss(nn.Module):
    """Combined detection loss with Hungarian matching.

    Computes:
        1. Focal classification loss on matched pairs + "no object" for unmatched.
        2. L1 bounding box regression loss on matched pairs.
        3. GIoU loss on matched pairs.
        4. (Optional) auxiliary losses from intermediate decoder layers.

    Args:
        num_classes: Number of foreground classes.
        matcher: HungarianMatcher instance.
        weight_ce: Classification loss weight.
        weight_bbox: L1 bbox loss weight.
        weight_giou: GIoU loss weight.
        eos_coef: Relative weight of the "no-object" class in CE loss.
            Lower values down-weight the dominant background class.
        focal_alpha: Alpha for focal loss.
        focal_gamma: Gamma for focal loss.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_ce: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        eos_coef: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_ce = weight_ce
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute all detection losses.

        Args:
            outputs: Dict with 'pred_logits', 'pred_boxes', and
                optional 'aux_outputs'.
            targets: List[B] of dicts with 'labels' and 'boxes'.

        Returns:
            Dict with 'loss' (total), 'loss_ce', 'loss_bbox', 'loss_giou',
            and per-layer aux keys if present.
        """
        indices = self.matcher(outputs, targets)

        loss_ce = self._loss_classification(outputs, targets, indices)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices)

        total = (
            self.weight_ce * loss_ce
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou
        )

        loss_dict = {
            "loss": total,
            "loss_ce": loss_ce.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
        }

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets)
                aux_ce = self._loss_classification(aux, targets, aux_indices)
                aux_bbox, aux_giou = self._loss_boxes(aux, targets, aux_indices)
                aux_loss = (
                    self.weight_ce * aux_ce
                    + self.weight_bbox * aux_bbox
                    + self.weight_giou * aux_giou
                )
                total = total + aux_loss
                loss_dict[f"loss_ce_{i}"] = aux_ce.detach()
                loss_dict[f"loss_bbox_{i}"] = aux_bbox.detach()
                loss_dict[f"loss_giou_{i}"] = aux_giou.detach()

            loss_dict["loss"] = total

        return loss_dict

    def _loss_classification(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        logits = outputs["pred_logits"]  # (B, Q, C+1)
        B, Q, _ = logits.shape

        target_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.long, device=logits.device
        )
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][tgt_idx]

        loss = F.cross_entropy(
            logits.reshape(-1, self.num_classes + 1),
            target_classes.reshape(-1),
            weight=self.empty_weight,
        )
        return loss

    def _loss_boxes(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        pred_boxes = outputs["pred_boxes"]
        device = pred_boxes.device

        src_boxes = []
        tgt_boxes = []
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                src_boxes.append(pred_boxes[b, pred_idx])
                tgt_boxes.append(targets[b]["boxes"][tgt_idx])

        if not src_boxes:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero

        src_boxes = torch.cat(src_boxes)
        tgt_boxes = torch.cat(tgt_boxes)

        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou = generalized_box_iou(src_xyxy, tgt_xyxy)
        loss_giou = (1 - giou.diag()).mean()

        return loss_l1, loss_giou

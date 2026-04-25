"""Detection losses: classification (focal, IA-BCE, or CE) + bbox (L1 + GIoU).

All loss weights and the classification loss type are injected via
constructor — nothing hardcoded.
Operates on matched prediction-target pairs produced by HungarianMatcher.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_iou

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
        1. Classification loss (focal or cross-entropy) on matched pairs.
        2. L1 bounding box regression loss on matched pairs.
        3. GIoU loss on matched pairs.
        4. (Optional) auxiliary losses from intermediate decoder layers.

    Args:
        num_classes: Number of foreground classes.
        matcher: HungarianMatcher instance.
        cls_type: ``"focal"`` for sigmoid focal loss (DETR v2 / DINOv3 Obj365
            stages), ``"ia_bce"`` for IoU-aware BCE (Align-DETR-style; DINOv3
            COCO fine-tune), or ``"cross_entropy"`` for softmax CE (original DETR).
        weight_cls: Classification loss weight.
        weight_bbox: L1 bbox loss weight.
        weight_giou: GIoU loss weight.
        eos_coef: Background class weight (cross_entropy only).
        focal_alpha: Alpha for focal loss.
        focal_gamma: Gamma for focal loss.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        cls_type: str = "focal",
        weight_cls: float = 2.0,
        weight_bbox: float = 1.0,
        weight_giou: float = 2.0,
        eos_coef: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        if cls_type not in ("focal", "ia_bce", "cross_entropy"):
            raise ValueError(
                f"cls_type must be 'focal', 'ia_bce', or 'cross_entropy', got '{cls_type}'"
            )

        self.num_classes = num_classes
        self.matcher = matcher
        self.cls_type = cls_type
        self.weight_cls = weight_cls
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

        loss_cls = self._loss_classification(outputs, targets, indices)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices)

        total = (
            self.weight_cls * loss_cls
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou
        )

        loss_dict = {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
        }

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets)
                aux_cls = self._loss_classification(aux, targets, aux_indices)
                aux_bbox, aux_giou = self._loss_boxes(aux, targets, aux_indices)
                aux_loss = (
                    self.weight_cls * aux_cls
                    + self.weight_bbox * aux_bbox
                    + self.weight_giou * aux_giou
                )
                total = total + aux_loss
                loss_dict[f"loss_cls_{i}"] = aux_cls.detach()
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

        if self.cls_type == "focal":
            return self._focal_classification(logits, targets, indices)
        if self.cls_type == "ia_bce":
            return self._ia_bce_classification(
                logits, outputs["pred_boxes"], targets, indices
            )
        return self._ce_classification(logits, targets, indices)

    def _ce_classification(
        self,
        logits: Tensor,
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Softmax cross-entropy classification (original DETR)."""
        B, Q, _ = logits.shape
        target_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.long, device=logits.device
        )
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][tgt_idx]

        return F.cross_entropy(
            logits.reshape(-1, self.num_classes + 1),
            target_classes.reshape(-1),
            weight=self.empty_weight,
        )

    def _focal_classification(
        self,
        logits: Tensor,
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Sigmoid focal loss classification (Deformable DETR / DINOv3).

        Uses per-class binary sigmoid rather than softmax, so the
        background class is implicit (all sigmoids near zero).
        Only the first ``num_classes`` logit channels are used.
        """
        B, Q, _ = logits.shape
        src_logits = logits[:, :, :self.num_classes]  # drop background channel

        target_onehot = torch.zeros(
            (B, Q, self.num_classes), dtype=src_logits.dtype, device=src_logits.device,
        )
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                cls_ids = targets[b]["labels"][tgt_idx]
                target_onehot[b, pred_idx, cls_ids] = 1.0

        num_boxes = max(sum(t["labels"].shape[0] for t in targets), 1)
        loss = sigmoid_focal_loss(
            src_logits.reshape(-1, self.num_classes),
            target_onehot.reshape(-1, self.num_classes),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        )
        return loss / num_boxes

    def _ia_bce_classification(
        self,
        logits: Tensor,
        pred_boxes: Tensor,
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """IoU-aware BCE (IA-BCE), simplified from Align-DETR (Cai et al.).

        One-to-one Hungarian matches only (no multi-candidate rank reweighting).
        ``focal_alpha`` is reused as the quality mixing exponent α in
        ``t = (p^α)(IoU^(1-α))`` (detached target on positives).
        ``focal_gamma`` is the negative exponent γ on ``p^γ`` as in Align-DETR.
        """
        B, Q, _ = logits.shape
        src_logits = logits[:, :, : self.num_classes]
        prob = src_logits.sigmoid()
        neg_weights = prob.pow(self.focal_gamma)
        pos_weights = torch.zeros_like(src_logits)

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            pb = pred_boxes[b, pred_idx]
            tb = targets[b]["boxes"][tgt_idx]
            pb_xy = box_cxcywh_to_xyxy(pb)
            tb_xy = box_cxcywh_to_xyxy(tb)
            ious = box_iou(pb_xy, tb_xy).diag()
            cls_ids = targets[b]["labels"][tgt_idx]
            alpha = self.focal_alpha
            for k in range(pred_idx.shape[0]):
                q = int(pred_idx[k].item())
                c = int(cls_ids[k].item())
                p = prob[b, q, c]
                iou = ious[k].clamp(0.0, 1.0)
                t = (p.detach().pow(alpha) * iou.pow(1.0 - alpha)).clamp(min=0.01)
                pos_weights[b, q, c] = t
                neg_weights[b, q, c] = 1.0 - t

        eps = 1e-8
        p = prob.clamp(eps, 1.0 - eps)
        per_elem = -pos_weights * p.log() - neg_weights * (1.0 - p).log()
        num_boxes = max(sum(t["labels"].shape[0] for t in targets), 1)
        return per_elem.sum() / num_boxes

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
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        src_boxes = torch.cat(src_boxes)
        tgt_boxes = torch.cat(tgt_boxes)

        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou = generalized_box_iou(src_xyxy, tgt_xyxy)
        loss_giou = (1 - giou.diag()).mean()

        return loss_l1, loss_giou

"""Hungarian matcher for bipartite assignment between predictions and ground truth.

Implements the optimal assignment used in DETR: for each image in a batch,
finds the permutation of predicted slots that minimizes a combined cost of
classification confidence, L1 bbox distance, and GIoU.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU between two sets of boxes in xyxy format.

    Args:
        boxes1: (N, 4) in xyxy.
        boxes2: (M, 4) in xyxy.

    Returns:
        (N, M) pairwise GIoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)

    enclosing_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_area = (enclosing_rb - enclosing_lt).clamp(min=0).prod(dim=2)

    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground truth.

    Computes a weighted cost matrix per image, then solves the
    linear assignment problem with `scipy.optimize.linear_sum_assignment`.

    The classification cost MUST share its probability space with the
    classification loss:

    - ``cls_type="focal"`` or ``"ia_bce"``  → sigmoid (per-class binary). The
      background channel is not used for cost (same as focal training).
    - ``cls_type="cross_entropy"``  → softmax over ``C+1`` channels (the
      original DETR formulation).

    Args:
        cost_class: Weight for classification cost.
        cost_bbox: Weight for L1 bounding box cost.
        cost_giou: Weight for GIoU cost.
        cls_type: ``"focal"``, ``"ia_bce"``, or ``"cross_entropy"`` — must match the loss.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cls_type: str = "focal",
    ):
        super().__init__()
        if cls_type not in ("focal", "ia_bce", "cross_entropy"):
            raise ValueError(
                f"cls_type must be 'focal', 'ia_bce', or 'cross_entropy', got '{cls_type}'"
            )
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cls_type = cls_type

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> List[Tuple[Tensor, Tensor]]:
        """Compute optimal bipartite matching for the batch.

        Args:
            outputs: Dict with 'pred_logits' (B, Q, C+1) and
                'pred_boxes' (B, Q, 4) in cxcywh.
            targets: List[B] of dicts, each with 'labels' (N,) and
                'boxes' (N, 4) in cxcywh.

        Returns:
            List[B] of (pred_indices, target_indices) tuples.
        """
        logits = outputs["pred_logits"]
        B, Q, C_plus_1 = logits.shape
        num_fg = C_plus_1 - 1

        if self.cls_type in ("focal", "ia_bce"):
            out_prob = logits[:, :, :num_fg].sigmoid()  # (B, Q, C)
        else:
            out_prob = logits.softmax(-1)               # (B, Q, C+1)

        out_bbox = outputs["pred_boxes"]  # (B, Q, 4)

        indices = []
        for b in range(B):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            if tgt_ids.numel() == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=out_prob.device),
                    torch.tensor([], dtype=torch.long, device=out_prob.device),
                ))
                continue

            cost_class = -out_prob[b, :, tgt_ids]  # (Q, N)
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)  # (Q, N)

            out_xyxy = box_cxcywh_to_xyxy(out_bbox[b])
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = -generalized_box_iou(out_xyxy, tgt_xyxy)  # (Q, N)

            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=out_prob.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=out_prob.device),
            ))

        return indices

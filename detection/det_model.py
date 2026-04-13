"""Detection model: composes a frozen backbone with a DETR detection head.

This is pure composition — swap any backbone that exposes
``(features, spatial_shape)`` and any head that accepts
``(features, spatial_shape)``.  The backbone type (``vit`` or ``cnn``)
is selected at build time; the rest of the pipeline is unchanged.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import FrozenVitBackbone, FrozenCnnBackbone
from .detr_head import DETRHead

_BACKBONE_REGISTRY = {
    "vit": FrozenVitBackbone,
    "cnn": FrozenCnnBackbone,
}


class DetectionModel(nn.Module):
    """Full detection model: frozen backbone + trainable DETR head.

    Args:
        backbone: Frozen feature extractor returning
            ``(B, H*W, D)`` tokens and ``(H, W)`` spatial shape.
        head: Detection head accepting ``(features, spatial_shape)``
            and returning prediction dicts.
    """

    def __init__(self, backbone: nn.Module, head: DETRHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features, spatial_shape = self.backbone(x)
        return self.head(features, spatial_shape)

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    def trainable_parameters(self):
        """Yield only the trainable (head) parameters for the optimizer."""
        return self.head.parameters()

    def trainable_named_parameters(self):
        """Yield named trainable parameters for param-group inspection."""
        return self.head.named_parameters()


def build_detection_model(
    backbone_type: str = "vit",
    model_name: str = "vit_base_patch16_rope_reg1_gap_256",
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Optional[int] = None,
    num_classes: int = 91,
    num_queries: int = 100,
    d_model: int = 256,
    nhead: int = 8,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    aux_loss: bool = True,
) -> DetectionModel:
    """Factory function — single call to wire backbone + head.

    Args:
        backbone_type: Backbone family — ``"vit"`` or ``"cnn"``.
        model_name: timm registry name for the backbone.
        pretrained: Load timm pretrained weights for backbone.
        checkpoint_path: Local checkpoint path for backbone weights.
        img_size: Override image size for the backbone.
        num_classes: Number of foreground detection classes.
        num_queries: Number of DETR object queries.
        d_model: Decoder hidden dimension.
        nhead: Decoder attention heads.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: Decoder FFN hidden dimension.
        dropout: Decoder dropout rate.
        aux_loss: Enable auxiliary losses from intermediate decoder layers.

    Returns:
        Assembled DetectionModel ready for training.
    """
    if backbone_type not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone type '{backbone_type}'. "
            f"Choose from: {list(_BACKBONE_REGISTRY.keys())}"
        )

    backbone_cls = _BACKBONE_REGISTRY[backbone_type]
    backbone = backbone_cls(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        img_size=img_size,
    )

    head = DETRHead(
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        num_queries=num_queries,
        dropout=dropout,
        backbone_dim=backbone.embed_dim,
        aux_loss=aux_loss,
    )

    return DetectionModel(backbone=backbone, head=head)

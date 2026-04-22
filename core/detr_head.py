"""DETR detection head: transformer decoder + classification/bbox FFN heads.

Implements the detection head from "End-to-End Object Detection with Transformers"
(Carion et al., 2020), adapted for ViT backbone token features following the
DINOv2/v3 approach of using a frozen ViT encoder with a lightweight decoder.

The ViT backbone already serves as the encoder, so this module only contains
the transformer decoder that cross-attends learnable object queries to the
encoded features, plus the prediction FFN heads.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .position_encoding import PositionEncoding2D


class MLP(nn.Module):
    """Multi-layer perceptron used for bbox regression.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        num_layers: Number of linear layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = torch.relu(x)
        return x


class DETRHead(nn.Module):
    """DETR-style detection head with transformer decoder.

    Takes encoded features from a ViT backbone (sequence of token embeddings),
    cross-attends learnable object queries to them, and produces per-query
    class logits and bounding box predictions.

    Args:
        d_model: Decoder hidden dimension and query embedding dimension.
        nhead: Number of attention heads in decoder layers.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: FFN hidden dimension inside decoder layers.
        num_classes: Number of foreground object classes (background is implicit).
        num_queries: Number of learnable object queries (max detections per image).
        dropout: Dropout rate in decoder layers.
        backbone_dim: Backbone output dimension. If != d_model, a linear
            projection is added.
        aux_loss: If True, return intermediate decoder layer outputs for
            auxiliary loss computation (DETR deep supervision).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        num_classes: int = 91,
        num_queries: int = 100,
        dropout: float = 0.1,
        backbone_dim: int = 768,
        aux_loss: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.aux_loss = aux_loss

        self.input_proj = (
            nn.Linear(backbone_dim, d_model)
            if backbone_dim != d_model
            else nn.Identity()
        )

        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm,
        )

        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = MLP(d_model, d_model, output_dim=4, num_layers=3)

        self.pos_encoder = PositionEncoding2D(d_model=d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_embed.weight)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        for layer in self.bbox_head.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        src: Tensor,
        spatial_shape: tuple,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass through the detection head.

        Args:
            src: Backbone token features, shape (B, N, backbone_dim).
                 N = H_feat * W_feat (prefix tokens should already be stripped).
            spatial_shape: (H_feat, W_feat) spatial grid dimensions.
            mask: Optional padding mask (B, H_feat, W_feat), True = padded.

        Returns:
            Dictionary with:
                - 'pred_logits': (B, num_queries, num_classes+1)
                - 'pred_boxes': (B, num_queries, 4) in normalized (cx, cy, w, h)
                - 'aux_outputs': list of dicts from intermediate layers (if aux_loss)
        """
        B = src.shape[0]
        h, w = spatial_shape

        memory = self.input_proj(src)  # (B, H*W, d_model)

        pos_embed = self.pos_encoder(B, h, w, device=src.device, dtype=src.dtype)
        memory = memory + pos_embed

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        if self.aux_loss:
            hs = self._forward_decoder_with_intermediates(queries, memory)
        else:
            hs = self.decoder(queries, memory).unsqueeze(0)  # (1, B, Q, d_model)

        outputs_class = self.class_head(hs)  # (L, B, Q, C+1)
        outputs_bbox = self.bbox_head(hs).sigmoid()  # (L, B, Q, 4)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_bbox[-1],
        }

        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": c, "pred_boxes": b}
                for c, b in zip(outputs_class[:-1], outputs_bbox[:-1])
            ]

        return out

    def _forward_decoder_with_intermediates(
        self,
        tgt: Tensor,
        memory: Tensor,
    ) -> Tensor:
        """Run decoder collecting intermediate outputs for auxiliary loss.

        Returns:
            Stacked outputs of shape (num_layers, B, Q, d_model).
        """
        output = tgt
        intermediates = []

        for layer in self.decoder.layers:
            output = layer(output, memory)
            intermediates.append(self.decoder.norm(output))

        return torch.stack(intermediates)  # (L, B, Q, d_model)

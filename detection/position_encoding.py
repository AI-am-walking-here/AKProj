"""Sinusoidal 2D position encoding for DETR-style detection heads.

Generates fixed (non-learnable) sinusoidal embeddings for spatial feature grids,
following the encoding scheme from "End-to-End Object Detection with Transformers"
(Carion et al., 2020).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PositionEncoding2D(nn.Module):
    """Fixed sinusoidal 2D position encoding.

    Produces (B, H*W, d_model) embeddings from spatial dimensions.
    Temperature-scaled sin/cos on normalized [0,1] grid coordinates,
    half channels for y-axis, half for x-axis.

    Args:
        d_model: Embedding dimension (must be even).
        temperature: Base temperature for frequency scaling.
        normalize: Whether to normalize grid coordinates to [0,1].
        scale: Scaling factor applied after normalization.
    """

    def __init__(
        self,
        d_model: int = 256,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = 2.0 * math.pi,
    ):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(
        self,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate 2D sinusoidal position embeddings.

        Args:
            batch_size: Batch dimension.
            h: Height of spatial grid.
            w: Width of spatial grid.
            device: Target device.
            dtype: Target dtype.
            mask: Optional (B, H, W) bool mask where True = padding.

        Returns:
            Position embeddings of shape (B, H*W, d_model).
        """
        if mask is not None:
            not_mask = ~mask
            y_embed = not_mask.cumsum(dim=1, dtype=dtype)
            x_embed = not_mask.cumsum(dim=2, dtype=dtype)
        else:
            y_embed = torch.arange(1, h + 1, device=device, dtype=dtype)
            y_embed = y_embed.view(1, h, 1).expand(batch_size, h, w)
            x_embed = torch.arange(1, w + 1, device=device, dtype=dtype)
            x_embed = x_embed.view(1, 1, w).expand(batch_size, h, w)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        half_d = self.d_model // 2
        dim_t = torch.arange(half_d, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half_d)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4)
        pos_x = pos_x.flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4)
        pos_y = pos_y.flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3)  # (B, H, W, d_model)
        return pos.flatten(1, 2)  # (B, H*W, d_model)

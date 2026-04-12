"""Frozen ViT backbone extractor.

Loads a timm Eva model, freezes all parameters, and exposes only
`forward_features()` with prefix tokens stripped — yielding a clean
(B, H*W, D) spatial token tensor ready for a downstream detection head.

Imports timm via the vendored pytorch-image-models tree. The caller
must ensure `pytorch-image-models/` is on `sys.path` (see det_model.py
or detection_train.py).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import timm


class FrozenVitBackbone(nn.Module):
    """Wraps a timm ViT (Eva) as a frozen feature extractor for detection.

    Loads the model via `timm.create_model`, freezes every parameter,
    and strips prefix tokens (register / CLS) from the output of
    `forward_features` so downstream heads receive pure spatial tokens.

    Args:
        model_name: timm model registry name
            (e.g. "vit_base_patch16_rope_reg1_gap_256").
        pretrained: Load timm pretrained weights (from HF hub).
        checkpoint_path: Path to a local `.pth` checkpoint file.
            Mutually exclusive with `pretrained` in practice —
            if both are set, `checkpoint_path` wins on the state_dict.
        img_size: Override image size (passed to timm factory).
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_rope_reg1_gap_256",
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        img_size: Optional[int] = None,
    ):
        super().__init__()

        factory_kwargs = {}
        if img_size is not None:
            factory_kwargs["img_size"] = img_size

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # strips the classification head
            **factory_kwargs,
        )

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.embed_dim: int = self.model.embed_dim
        self.num_prefix_tokens: int = getattr(self.model, "num_prefix_tokens", 0)

        grid_size = self.model.patch_embed.grid_size
        self.grid_size: Tuple[int, int] = (
            tuple(grid_size) if not isinstance(grid_size, tuple) else grid_size
        )

    def _load_checkpoint(self, path: str) -> None:
        """Load a classification checkpoint, dropping head-related keys."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        if "state_dict_ema" in state:
            state = state["state_dict_ema"]

        drop_prefixes = ("head.", "fc_norm.", "attn_pool.")
        state = {
            k: v for k, v in state.items()
            if not any(k.startswith(p) for p in drop_prefixes)
        }
        self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Extract spatial token features from the frozen backbone.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            features: (B, H_feat * W_feat, embed_dim) spatial tokens,
                prefix tokens stripped.
            spatial_shape: (H_feat, W_feat) grid dimensions.
        """
        tokens = self.model.forward_features(x)  # (B, num_prefix + H*W, D)
        spatial_tokens = tokens[:, self.num_prefix_tokens:, :]  # strip register/cls
        return spatial_tokens, self.grid_size

    def train(self, mode: bool = True) -> "FrozenVitBackbone":
        """Override to keep backbone permanently in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

"""Frozen backbone feature extractors for detection.

Provides both ViT and CNN backbone wrappers that share a common output
interface: ``(features, spatial_shape)`` where ``features`` is
``(B, H*W, D)`` and ``spatial_shape`` is ``(H, W)``.

Any detection head that consumes this contract works with either backbone,
enabling apples-to-apples comparison of backbone transfer quality.

Requires ``timm`` (pip install timm).
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
        state = torch.load(path, map_location="cpu", weights_only=False)
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


class FrozenCnnBackbone(nn.Module):
    """Wraps a timm CNN as a frozen feature extractor for detection.

    Extracts the final-stage spatial feature map via ``forward_features``,
    flattens it to ``(B, H*W, C)``, and returns the same
    ``(features, spatial_shape)`` tuple as ``FrozenVitBackbone`` so the
    downstream detection head is completely backbone-agnostic.

    Works with any timm CNN whose ``forward_features`` returns a 4-D
    ``(B, C, H, W)`` tensor: ResNet, ConvNeXt, EfficientNet, etc.

    Args:
        model_name: timm model registry name (e.g. ``"resnet50"``).
        pretrained: Load timm pretrained weights (from HF hub).
        checkpoint_path: Path to a local ``.pth`` checkpoint file.
        img_size: Override image size (passed to timm factory).
    """

    def __init__(
        self,
        model_name: str = "resnet50",
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
            num_classes=0,
            **factory_kwargs,
        )

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.embed_dim: int = self.model.num_features

        self._img_size = img_size or 256
        self.grid_size: Tuple[int, int] = self._infer_grid_size()

    def _infer_grid_size(self) -> Tuple[int, int]:
        """Run a dummy forward pass to determine the spatial grid size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self._img_size, self._img_size)
            feat = self.model.forward_features(dummy)
        if feat.dim() == 4:
            return (feat.shape[2], feat.shape[3])
        raise ValueError(
            f"Expected 4-D (B,C,H,W) from CNN forward_features, "
            f"got shape {feat.shape}. Model '{type(self.model).__name__}' "
            f"may not be a standard CNN."
        )

    def _load_checkpoint(self, path: str) -> None:
        """Load a classification checkpoint, dropping head-related keys."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        if "state_dict_ema" in state:
            state = state["state_dict_ema"]

        drop_prefixes = ("head.", "fc.", "classifier.")
        state = {
            k: v for k, v in state.items()
            if not any(k.startswith(p) for p in drop_prefixes)
        }
        self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Extract spatial features from the frozen CNN backbone.

        Args:
            x: Input images ``(B, C, H, W)``.

        Returns:
            features: ``(B, H_feat * W_feat, embed_dim)`` spatial features.
            spatial_shape: ``(H_feat, W_feat)`` grid dimensions.
        """
        feat = self.model.forward_features(x)  # (B, C, H, W)
        B, C, H, W = feat.shape
        spatial_tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return spatial_tokens, (H, W)

    def train(self, mode: bool = True) -> "FrozenCnnBackbone":
        """Override to keep backbone permanently in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

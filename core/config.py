"""YAML config loader with CLI override support.

Load a YAML config file, then let argparse flags override individual
values so you can do::

    python detection_train.py --config configs/default.yaml --lr 1e-4

The ``Config`` wrapper gives dot-access (``cfg.backbone.name``) while
remaining a plain nested dict underneath.
"""
import argparse
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class Config:
    """Thin wrapper that turns a nested dict into attribute-accessible objects."""

    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _set_nested(d: Dict, dotted_key: str, value: Any) -> None:
    """Set ``d["a"]["b"]`` given ``dotted_key="a.b"``."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# Maps flat CLI flag → dotted YAML path.
_CLI_TO_YAML = {
    "config":           None,  # consumed before reaching YAML
    "backbone_type":    "backbone.type",
    "train_img_dir":    "data.train_img_dir",
    "train_ann":        "data.train_ann",
    "val_img_dir":      "data.val_img_dir",
    "val_ann":          "data.val_ann",
    "coco_o_img_dir":   "data.coco_o_img_dir",
    "coco_o_ann":       "data.coco_o_ann",
    "img_size":         "backbone.img_size",
    "backbone":         "backbone.name",
    "pretrained":       "backbone.pretrained",
    "checkpoint":       "backbone.checkpoint",
    "num_classes":      "data.num_classes",
    "num_queries":      "head.num_queries",
    "d_model":          "head.d_model",
    "nhead":            "head.nhead",
    "num_decoder_layers": "head.num_decoder_layers",
    "dim_feedforward":  "head.dim_feedforward",
    "dropout":          "head.dropout",
    "no_aux_loss":      "head.aux_loss",       # inverted bool
    "cls_type":         "loss.cls_type",
    "weight_cls":       "loss.weight_cls",
    "weight_bbox":      "loss.weight_bbox",
    "weight_giou":      "loss.weight_giou",
    "eos_coef":         "loss.eos_coef",
    "cost_class":       "matcher.cost_class",
    "cost_bbox":        "matcher.cost_bbox",
    "cost_giou":        "matcher.cost_giou",
    "epochs":           "training.epochs",
    "batch_size":       "training.batch_size",
    "lr":               "training.lr",
    "weight_decay":     "training.weight_decay",
    "lr_drop":          "training.lr_drop",
    "max_grad_norm":    "training.max_grad_norm",
    "num_workers":      "training.num_workers",
    "amp":              "training.amp",
    "no_amp":           "training.amp",        # inverted bool
    "resume":           "training.resume",
    "eval_interval":    "eval.interval",
    "output_dir":       "output.dir",
    "log_interval":     "output.log_interval",
    "save_interval":    "output.save_interval",
    "device":           "device",
}


def build_cli_parser() -> argparse.ArgumentParser:
    """Minimal argparse that mirrors the YAML keys for override convenience."""
    p = argparse.ArgumentParser(
        description="DETR detection training with frozen ViT backbone",
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file")

    # data
    p.add_argument("--train-img-dir", type=str, default=None)
    p.add_argument("--train-ann", type=str, default=None)
    p.add_argument("--val-img-dir", type=str, default=None)
    p.add_argument("--val-ann", type=str, default=None)
    p.add_argument("--coco-o-img-dir", type=str, default=None)
    p.add_argument("--coco-o-ann", type=str, default=None)
    p.add_argument("--img-size", type=int, default=None)

    # backbone
    p.add_argument("--backbone-type", type=str, default=None,
                   choices=["vit", "cnn"],
                   help="Backbone family: vit or cnn")
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--pretrained", action="store_true", default=None)
    p.add_argument("--checkpoint", type=str, default=None)

    # head
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--num-queries", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--num-decoder-layers", type=int, default=None)
    p.add_argument("--dim-feedforward", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--no-aux-loss", action="store_true", default=None)

    # loss
    p.add_argument("--cls-type", type=str, default=None,
                   choices=["focal", "cross_entropy"])
    p.add_argument("--weight-cls", type=float, default=None)
    p.add_argument("--weight-bbox", type=float, default=None)
    p.add_argument("--weight-giou", type=float, default=None)
    p.add_argument("--eos-coef", type=float, default=None)

    # matcher
    p.add_argument("--cost-class", type=float, default=None)
    p.add_argument("--cost-bbox", type=float, default=None)
    p.add_argument("--cost-giou", type=float, default=None)

    # training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--lr-drop", type=int, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--amp", action="store_true", default=None,
                   help="Enable mixed precision training")
    p.add_argument("--no-amp", action="store_true", default=None,
                   help="Disable mixed precision training")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from")

    # eval
    p.add_argument("--eval-interval", type=int, default=None)

    # output
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--save-interval", type=int, default=None)

    # device
    p.add_argument("--device", type=str, default=None)

    return p


def load_config(argv: Optional[list] = None) -> Config:
    """Load YAML config, apply CLI overrides, return ``Config`` object.

    Priority: CLI flag > YAML file > built-in defaults below.
    """
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    defaults: Dict[str, Any] = {
        "backbone": {"type": "vit", "name": "vit_base_patch16_rope_reg1_gap_256",
                      "checkpoint": None, "pretrained": False, "img_size": 256},
        "head": {"type": "detr", "d_model": 256, "nhead": 8, "num_decoder_layers": 6,
                 "dim_feedforward": 2048, "dropout": 0.1, "num_queries": 100, "aux_loss": True},
        "data": {"train_img_dir": None, "train_ann": None, "val_img_dir": None,
                 "val_ann": None, "coco_o_img_dir": None, "coco_o_ann": None, "num_classes": None},
        "augmentation": {"horizontal_flip": True, "color_jitter": False,
                         "multiscale": None, "random_crop": False, "crop_min_scale": 0.5},
        "loss": {"cls_type": "focal", "focal_alpha": 0.25, "focal_gamma": 2.0,
                 "weight_cls": 2.0, "weight_bbox": 1.0, "weight_giou": 2.0, "eos_coef": 0.1},
        "matcher": {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        "training": {"epochs": 50, "batch_size": 4, "lr": 5e-5, "weight_decay": 0.05,
                      "lr_schedule": "step", "lr_drop": 40, "warmup_steps": 1000,
                      "max_grad_norm": 0.1, "num_workers": 4, "amp": True,
                      "resume": None},
        "eval": {"interval": 1, "score_threshold": 0.01, "max_detections": 100},
        "output": {"dir": "output/detection", "log_interval": 50, "save_interval": 5},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if args.config is not None:
        yaml_cfg = load_yaml(args.config)
        _deep_merge(defaults, yaml_cfg)

    cli_dict = vars(args)
    for cli_key, yaml_path in _CLI_TO_YAML.items():
        if yaml_path is None:
            continue
        val = cli_dict.get(cli_key)
        if val is None:
            continue
        if cli_key in ("no_aux_loss", "no_amp"):
            val = not val
        _set_nested(defaults, yaml_path, val)

    return Config(defaults)


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v

"""Standalone evaluation script — frozen backbone + DETR head.

Loads a trained detection head checkpoint, runs it against any COCO-format
dataset, and writes paper-ready metrics (AP, AP50, AP75, AP_S/M/L + AR
variants) to a JSON results file.

Designed to be completely independent of detection_train.py so it can be
handed to any AI or team member without needing to understand the training loop.

Usage
-----
# Evaluate ViT backbone on COCO val2017:
python evaluate.py \\
    --config configs/default.yaml \\
    --checkpoint output/detection/best.pth \\
    --train-ann data/objects365/annotations/train.json \\
    --ann-file data/coco/annotations/instances_val2017.json \\
    --img-dir data/coco/val2017 \\
    --output results/vit_coco.json \\
    --model-name "ViT-B DETR" \\
    --per-class

# Evaluate without cross-dataset remapping (eval on same domain as training):
python evaluate.py \\
    --config configs/default.yaml \\
    --checkpoint output/detection/best.pth \\
    --ann-file data/objects365/annotations/val.json \\
    --img-dir data/objects365/val \\
    --output results/vit_obj365.json
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from core.telemetry import build_sink

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg,
    num_classes: int,
    device: torch.device,
):
    """Build detection model and load the trained head from *checkpoint_path*.

    The backbone is always rebuilt from scratch (frozen, no gradient).
    Only the DETR head weights are loaded from the checkpoint.

    Args:
        checkpoint_path: Path to a ``.pth`` file containing
            ``head_state_dict`` (saved by ``detection_train.py``).
        cfg: Config object (from ``detection.config.load_config``).
        num_classes: Number of foreground classes the head was trained with.
            Must match the checkpoint's head dimensions.
        device: Device to load the model onto.

    Returns:
        DetectionModel in eval mode with head weights restored.
    """
    from core.det_model import build_detection_model

    model = build_detection_model(
        backbone_type=cfg.backbone.type,
        model_name=cfg.backbone.name,
        pretrained=False,
        checkpoint_path=cfg.backbone.checkpoint,
        img_size=cfg.backbone.img_size,
        num_classes=num_classes,
        num_queries=cfg.head.num_queries,
        d_model=cfg.head.d_model,
        nhead=cfg.head.nhead,
        num_decoder_layers=cfg.head.num_decoder_layers,
        dim_feedforward=cfg.head.dim_feedforward,
        dropout=cfg.head.dropout,
        aux_loss=False,  # aux_loss not needed at inference
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "head_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' has no 'head_state_dict' key. "
            f"Keys found: {list(ckpt.keys())}"
        )
    model.head.load_state_dict(ckpt["head_state_dict"], strict=True)
    model.eval()
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    _logger.info(f"Model loaded: {total:,} parameters, head from {checkpoint_path}")
    return model


# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #

def build_eval_dataloader(
    img_dir: str,
    ann_file: str,
    cfg,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    """Build a shuffle=False dataloader over an eval dataset.

    Args:
        img_dir: Directory containing images.
        ann_file: COCO-format annotation JSON for the eval split.
        cfg: Config object used to read ``img_size``.
        batch_size: Batch size for inference (can be larger than training).
        num_workers: DataLoader worker count.

    Returns:
        DataLoader over the eval set, no augmentation.
    """
    from core.datasets import CocoFormatDataset, collate_fn
    from core.transforms import build_transforms

    eval_transform = build_transforms(img_size=cfg.backbone.img_size, training=False)

    dataset = CocoFormatDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        img_size=cfg.backbone.img_size,
        transform=eval_transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# --------------------------------------------------------------------------- #
# Class mapping helpers
# --------------------------------------------------------------------------- #

def build_eval_mapping(
    train_ann: Optional[str],
    eval_ann: str,
) -> Tuple[Optional[Dict[int, int]], Dict[int, int], int]:
    """Build source→target label mapping for cross-dataset evaluation.

    Args:
        train_ann: Source (training) annotation file. If ``None``, the eval
            dataset's own category ordering is used directly (no remapping).
        eval_ann: Target (evaluation) annotation file.

    Returns:
        source_to_target: ``{src_label_idx: tgt_label_idx}`` or ``None``.
        target_label_to_cat_id: ``{tgt_label_idx: coco_cat_id}``.
        num_classes: Number of source classes (for building the model head).
    """
    import json

    if train_ann is None:
        _logger.info("No --train-ann provided — evaluating without cross-dataset remapping")
        with open(eval_ann) as f:
            cats = json.load(f)["categories"]
        target_label_to_cat_id = {i: cat["id"] for i, cat in enumerate(cats)}
        num_classes = len(cats)
        return None, target_label_to_cat_id, num_classes

    from core.class_mapping import build_category_mapping
    import json

    with open(train_ann) as f:
        src_cats = json.load(f)["categories"]
    num_classes = len(src_cats)

    source_to_target, target_label_to_cat_id, unmatched = build_category_mapping(
        source_ann_file=train_ann,
        target_ann_file=eval_ann,
    )
    if unmatched:
        _logger.info(f"Unmatched source categories ({len(unmatched)}): {unmatched[:10]}...")

    return source_to_target, target_label_to_cat_id, num_classes


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_full_evaluation(
    model,
    loader: DataLoader,
    ann_file: str,
    cfg,
    source_to_target: Optional[Dict[int, int]],
    target_label_to_cat_id: Dict[int, int],
    per_class: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Run the model over *loader* and compute all metrics.

    Args:
        model: DetectionModel in eval mode.
        loader: Eval DataLoader (no augmentation).
        ann_file: Ground-truth annotation JSON for pycocotools.
        cfg: Config object (reads ``loss.cls_type``, ``eval.*``).
        source_to_target: Optional cross-dataset label remap.
        target_label_to_cat_id: Target label → COCO category ID.
        per_class: If ``True``, also compute per-category AP.
        device: Torch device for inference.

    Returns:
        Dict with standard 12 COCO metrics, and optionally
        ``"per_class_ap"`` sub-dict.
    """
    from core.coco_eval import predictions_to_coco_results
    from core.metrics import run_coco_evaluation, compute_per_class_ap

    cls_type = getattr(cfg.loss, "cls_type", "focal")
    score_threshold = getattr(cfg.eval, "score_threshold", 0.01)
    max_detections = getattr(cfg.eval, "max_detections", 100)

    model.eval()
    all_results: List[Dict] = []

    for batch_idx, (images, targets) in enumerate(loader):
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
            source_to_target_label=source_to_target,
            score_threshold=score_threshold,
            max_detections=max_detections,
            cls_type=cls_type,
        )
        all_results.extend(batch_results)

        if (batch_idx + 1) % 50 == 0:
            _logger.info(f"  Processed {batch_idx + 1}/{len(loader)} batches")

    _logger.info(f"Collected {len(all_results)} predictions — running COCO eval")
    metrics = run_coco_evaluation(all_results, ann_file)

    if per_class:
        _logger.info("Computing per-class AP breakdown ...")
        metrics["per_class_ap"] = compute_per_class_ap(all_results, ann_file)

    return metrics


# --------------------------------------------------------------------------- #
# Results persistence
# --------------------------------------------------------------------------- #

def save_results(
    metrics: Dict,
    model_name: str,
    backbone_type: str,
    checkpoint_path: str,
    dataset: str,
    ann_file: str,
    output_path: str,
) -> None:
    """Persist metrics + metadata to a JSON file.

    Args:
        metrics: Dict from ``run_full_evaluation``.
        model_name: Human-readable model label (used in the paper table).
        backbone_type: ``"vit"`` or ``"cnn"``.
        checkpoint_path: Path to the checkpoint that was evaluated.
        dataset: Short dataset name (e.g. ``"coco"``, ``"coco-o"``).
        ann_file: Annotation file path (for reproducibility).
        output_path: Destination ``.json`` file path.
    """
    record = {
        "metadata": {
            "model_name": model_name,
            "backbone_type": backbone_type,
            "checkpoint": str(checkpoint_path),
            "dataset": dataset,
            "ann_file": str(ann_file),
            "evaluated_at": datetime.datetime.now().isoformat(),
        },
        "metrics": metrics,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)

    _logger.info(f"Results saved to {out}")


# --------------------------------------------------------------------------- #
# Summary printer
# --------------------------------------------------------------------------- #

def print_summary(metrics: Dict, model_name: str) -> None:
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 60)
    print(f"  Results: {model_name}")
    print("=" * 60)
    core = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    ar = ["AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large"]
    for name, keys in [("Detection AP", core), ("Recall AR", ar)]:
        row = "  ".join(f"{k}={metrics.get(k, 0.0)*100:.1f}" for k in keys)
        print(f"  {name}:  {row}")
    if "per_class_ap" in metrics:
        pc = metrics["per_class_ap"]
        top5 = sorted(pc.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top-5 classes: " + ", ".join(f"{n}={v*100:.1f}" for n, v in top5))
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained detection head checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Path to YAML config (same one used for training)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint (must contain head_state_dict)")
    p.add_argument("--ann-file", required=True,
                   help="COCO-format annotation JSON for the eval dataset")
    p.add_argument("--img-dir", required=True,
                   help="Directory containing eval images")
    p.add_argument("--train-ann", default=None,
                   help="Training annotation JSON (needed for cross-dataset label remapping). "
                        "Omit only if evaluating on the same domain as training.")
    p.add_argument("--output", required=True,
                   help="Output .json file path for metrics + metadata")
    p.add_argument("--model-name", default="Detection Model",
                   help="Human-readable name for this model (used in paper tables)")
    p.add_argument("--backbone-type", default=None,
                   help="Override backbone type from config (vit/cnn)")
    p.add_argument("--dataset", default="coco",
                   help="Short dataset name for metadata (e.g. coco, coco-o, objects365)")
    p.add_argument("--per-class", action="store_true",
                   help="Also compute per-category AP breakdown")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None,
                   help="Device override (cuda/cpu). Defaults to config value.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Load config (no training-specific CLI flags needed here)
    from core.config import load_yaml, Config, _deep_merge
    import yaml

    defaults = {
        "backbone": {"type": "vit", "name": "vit_base_patch16_rope_reg1_gap_256",
                     "checkpoint": None, "pretrained": False, "img_size": 256},
        "head": {"type": "detr", "d_model": 256, "nhead": 8, "num_decoder_layers": 6,
                 "dim_feedforward": 2048, "dropout": 0.1, "num_queries": 100, "aux_loss": True},
        "loss": {"cls_type": "focal"},
        "eval": {"score_threshold": 0.01, "max_detections": 100},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    _deep_merge(defaults, yaml_cfg)

    if args.backbone_type is not None:
        defaults["backbone"]["type"] = args.backbone_type
    if args.device is not None:
        defaults["device"] = args.device

    cfg = Config(defaults)
    device = torch.device(cfg.device)
    _logger.info(f"Device: {device}")

    sink = build_sink(cfg)

    # Class mapping + num_classes
    source_to_target, target_label_to_cat_id, num_classes = build_eval_mapping(
        train_ann=args.train_ann,
        eval_ann=args.ann_file,
    )
    _logger.info(f"num_classes (from training domain): {num_classes}")

    # Model
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        num_classes=num_classes,
        device=device,
    )

    # DataLoader
    loader = build_eval_dataloader(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _logger.info(f"Eval set: {len(loader.dataset)} images, {len(loader)} batches")

    # Evaluate
    metrics = run_full_evaluation(
        model=model,
        loader=loader,
        ann_file=args.ann_file,
        cfg=cfg,
        source_to_target=source_to_target,
        target_label_to_cat_id=target_label_to_cat_id,
        per_class=args.per_class,
        device=device,
    )
    sink.log_metrics({k: float(v) for k, v in metrics.items() if k != "per_class_ap"}, namespace="eval")
    if args.per_class and "per_class_ap" in metrics:
        # Log per-class AP as a table (category, AP)
        per_class_rows = [{"category": k, "AP": float(v)} for k, v in metrics["per_class_ap"].items()]
        sink.log_table("per_class_ap", per_class_rows)

    # Save
    save_results(
        metrics=metrics,
        model_name=args.model_name,
        backbone_type=cfg.backbone.type,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        ann_file=args.ann_file,
        output_path=args.output,
    )

    print_summary(metrics, args.model_name)
    sink.finish()


if __name__ == "__main__":
    main()

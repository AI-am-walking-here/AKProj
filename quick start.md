# Quick Start (AKProj — Frozen ViT + DETR Detection)

This repo fine-tunes a **DETR-style decoder head** on top of a **frozen timm backbone** (default: a ViT pretrained checkpoint), trained on **Objects365** and evaluated on **COCO / COCO-O**.

## What’s in this repo (full tree)

```text
AKProj/
  .gitattributes
  .gitignore
  README.md
  TODO.txt
  requirements.txt
  download_data.py
  detection_train.py
  configs/
    default.yaml
  detection/
    __init__.py
    backbone.py
    class_mapping.py
    coco_eval.py
    config.py
    datasets.py
    det_model.py
    detr_head.py
    losses.py
    matcher.py
    position_encoding.py
    transforms.py
    __pycache__/
      *.pyc
```

## Important “expected but missing” folders

These are **intentionally not committed** and are ignored by `.gitignore`, so they may not exist until you run training or download assets.

- **`data/`**: Dataset root (COCO / Objects365 paths are configured to live under here).
- **`checkpoints/`**: Pretrained backbone weights (e.g. `checkpoints/labelmix/model_best.pth.tar`).
- **`output/`**: Training outputs (best checkpoint + periodic resume checkpoints).
- **`pytorch-image-models/`**: Optional vendored copy of timm’s old repo; this codebase expects you to install `timm` via pip instead.

## How to run (minimal smoke test)

1) Install dependencies.

```bash
pip install -r requirements.txt
```

2) Download COCO val2017 (for evaluation smoke-testing).

```bash
python download_data.py --dataset coco
```

3) Train (requires Objects365 images + annotations; see below), optionally evaluate on COCO if you have COCO paths set.

```bash
python detection_train.py --config configs/default.yaml
```

## File-by-file: what each thing does (2 sentences each)

### Root files

- **`.gitattributes`**: Normalizes text files with auto line-ending handling so the repo behaves consistently across OSes. It does not affect runtime code—only how git stores/checks out text.
- **`.gitignore`**: Excludes large/generated directories like `data/`, `checkpoints/`, `output/`, and `pytorch-image-models/` from version control. This keeps the repo lightweight and forces datasets/weights to be managed locally.
- **`README.md`**: High-level overview of the architecture (frozen backbone → DETR head) and intended datasets (Objects365 training, COCO/COCO-O eval). It also documents the expected checkpoint path and the training command-line usage.
- **`TODO.txt`**: Project progress notes and design intent for features like augmentation, AMP, and resume behavior. It also lists a few remaining integration tasks (e.g., plumbing eval thresholds and using warmup steps).
- **`requirements.txt`**: Lists the Python dependencies required to run training/evaluation (PyTorch, torchvision, timm, scipy, pycocotools, Pillow, PyYAML). Versions are expressed as minimums so you can upgrade as needed.
- **`download_data.py`**: Downloads COCO val2017 images/annotations into `data/coco/` for a quick end-to-end pipeline test, with resume + retry logic. It can also stream a small sample of Objects365 **annotations only** from HuggingFace and writes a COCO-format JSON under `data/objects365/annotations/`.
- **`detection_train.py`**: Main training entry point that loads YAML/CLI config, builds datasets/transforms, constructs the model, and runs the train/eval loop. It saves `output/detection/best.pth` on best COCO AP and periodic resume checkpoints like `checkpoint_epoch_*.pth`.

### `configs/`

- **`configs/default.yaml`**: Default experiment wiring for backbone, head, data paths, augmentation toggles, matcher/loss weights, evaluation settings, and output directory. You can clone and edit this file to run experiments without touching code, and CLI flags override any YAML value.

### `detection/` (the modular detection library)

- **`detection/__init__.py`**: Re-exports the public “building blocks” (backbones, head, dataset, loss, matcher, eval, config loader) so `detection_train.py` can import from `detection` cleanly. Treat this as the stable API surface for swapping components during experiments.
- **`detection/backbone.py`**: Implements frozen backbone wrappers around timm models (ViT and CNN) that output a shared contract: `(B, H*W, D)` features plus `(H, W)` spatial shape. It freezes parameters and forces eval mode so only the detection head trains.
- **`detection/detr_head.py`**: Implements a DETR-style **decoder-only** head that cross-attends learnable queries to the frozen backbone features and predicts class logits + normalized boxes. It also optionally returns intermediate-layer predictions (`aux_outputs`) for auxiliary losses.
- **`detection/position_encoding.py`**: Generates fixed 2D sinusoidal positional embeddings for a `(H, W)` feature grid, returning a `(B, H*W, d_model)` tensor. The DETR head adds these embeddings to backbone features to inject spatial information.
- **`detection/det_model.py`**: Defines the composed `DetectionModel` (`backbone + head`) and a factory `build_detection_model(...)` that wires them together from config values. This keeps the rest of the pipeline backbone-agnostic and makes swapping backbones/head settings a one-call change.
- **`detection/datasets.py`**: Implements `CocoFormatDataset`, which reads a COCO-style JSON and returns resized/normalized images plus targets containing normalized `cxcywh` boxes and integer labels. It works across Objects365/COCO/COCO-O and provides a `collate_fn` for batching variable-length targets.
- **`detection/transforms.py`**: Detection-aware transforms that operate on `(image, target)` pairs so spatial ops like flip/crop remain label-consistent, and are toggled via an injected config dict. It builds separate pipelines for training (optional aug + normalize) and eval (resize + normalize only).
- **`detection/matcher.py`**: Implements Hungarian matching to assign predicted queries to ground-truth boxes per image using a weighted combination of class cost, L1 bbox cost, and GIoU cost. It uses `scipy.optimize.linear_sum_assignment`, which solves the assignment on CPU after building the cost matrix.
- **`detection/losses.py`**: Computes the detection loss on matched pairs: classification (either sigmoid focal loss or softmax cross-entropy), plus L1 and GIoU box losses. If `aux_outputs` are present it repeats the same loss per decoder layer and adds them into the total.
- **`detection/class_mapping.py`**: Builds a mapping between category sets by name (e.g., Objects365 → COCO) using case-insensitive matching and an alias table for known mismatches. This lets you train on a large-label dataset but evaluate on COCO’s 80 categories by dropping/unmapping classes.
- **`detection/coco_eval.py`**: Converts model outputs into COCO result dicts (absolute `xywh` + score + COCO category IDs), with optional source→target label remapping. It then runs COCO mAP via `pycocotools` and returns a metrics dictionary (AP, AP50, AP75, etc.).
- **`detection/config.py`**: Loads a YAML config and applies CLI overrides while providing dot-access via a thin `Config` wrapper. This keeps experiments “config-first” and avoids hardcoding hyperparameters or paths in the training code.
- **`detection/__pycache__/`**: Python bytecode cache created automatically when you run scripts or import modules. It is not source-of-truth and can be deleted safely; it will regenerate on the next run.

## Where the data and weights are supposed to live

- **Objects365 training** (expected by `configs/default.yaml`):
  - Images: `data/objects365/train/`
  - Annotations: `data/objects365/annotations/train.json`
  - Note: `download_data.py --dataset objects365` only creates the annotation JSON for a small streamed sample; you must acquire images separately for real training.
- **COCO evaluation**:
  - Images: `data/coco/val2017/`
  - Annotations: `data/coco/annotations/instances_val2017.json`
- **Backbone checkpoint**:
  - `checkpoints/labelmix/model_best.pth.tar` (path is configurable via YAML/CLI).


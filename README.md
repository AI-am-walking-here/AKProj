# Frozen ViT + DETR Object Detection

Fine-tunes a DETR-style detection head on top of a frozen ViT backbone (`vit_base_patch16_rope_reg1_gap_256`) pretrained on ImageNet-1k via the labelmix pipeline.

**Training data:** Objects365  
**Evaluation data:** COCO val, COCO-O

## Repo Structure (start here)

```
AKProj/
├─ __pycache__/
├─ checkpoints/
│  ├─ base/                      # backbone weights (ignored by git)
│  │  ├─ vit/
│  │  └─ cnn/
│  └─ trained/                   # trained head checkpoints + resume states (ignored by git)
│     ├─ vit/
│     └─ cnn/
│
├─ configs/
│  ├─ default.yaml               # main training config (paths, backbone, output, wandb)
│  └─ sweep.yaml                 # sweep config (which checkpoints/configs to evaluate)
│
├─ core/                         # reusable library code (many scripts live here)
│  ├─ __pycache__/
│  ├─ metrics/                   # COCO mAP computation + per-class AP + paper-ready table generation
│  ├─ telemetry/                 # optional logging adapters (W&B) + NullSink, used by train/eval to record metrics
│  ├─ backbone.py
│  ├─ det_model.py
│  ├─ detr_head.py
│  ├─ datasets.py
│  ├─ losses.py
│  └─ transforms.py
│
├─ data/
│  ├─ coco/                      # COCO val2017 (ignored by git)
│  ├─ coco-o/                    # COCO-O subset (ignored by git)
│  ├─ objects365/                # Objects365 subset (ignored by git)
│  ├─ download_coco.py           # COCO val2017 (5000 images + instances_val2017.json)
│  ├─ download_coco_o.py         # COCO-O subset (takes --zip path to manual download)
│  └─ download_objects365.py     # Objects365 subset (patch0 images + HF annotations)
│
├─ evaluations/
│  ├─ __pycache__/
│  ├─ __init__.py
│  ├─ evaluate.py                # single-model evaluation runner
│  └─ sweep_eval.py              # multi-model sweep + combined paper table
│
├─ pytorch-image-models/         # optional vendored timm tree (ignored by git)
│
├─ .gitattributes
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ train.py                      # training entrypoint (frozen backbone + DETR head)
```

## Architecture

```
Image (B, 3, 256, 256)
  → FrozenVitBackbone  [all params frozen, eval mode]
  → (B, H*W, 768) spatial tokens
  → DETRHead           [trainable decoder-only]
      input_proj (768 → 256)
      + 2D sinusoidal position encoding
      → TransformerDecoder (6 layers, 8 heads)
      → class_head (256 → num_classes+1)
      → bbox_head  (256 → 4, sigmoid → normalized cxcywh)
  → {pred_logits, pred_boxes, aux_outputs}
```

Only the DETR head is optimized. The ViT encoder is never updated.

## Checkpoint

| Field | Value |
|-------|-------|
| Model | `vit_base_patch16_rope_reg1_gap_256` |
| Dataset | ImageNet-1k |
| Best epoch | 107 |
| Top-1 | 82.05% |
| Top-5 | 96.10% |
| Path | `checkpoints/base/vit/model_best.pth.tar` |

## Pipeline

### Training (Objects365)

The model trains with `num_classes=365` (auto-detected from Objects365 annotations). The full ViT backbone is frozen; only the DETR decoder head is updated.

### Evaluation (COCO / COCO-O)

At eval time, `build_category_mapping` matches Objects365 categories to COCO's 80 categories by name (case-insensitive, with an alias table for known mismatches like `"Stuffed Toy" → "teddy bear"`). Predictions for unmapped classes are dropped. COCO mAP is computed via `pycocotools`.

COCO-O uses the same 80 COCO categories, so the same class mapping applies.

## Data Download

Each dataset has its own self-contained script. All three target ~5000 samples.

```bash
# COCO val2017 (evaluation) — downloads images + instances_val2017.json
python data/download_coco.py

# COCO-O (evaluation) — download the zip manually from alibaba/easyrobust
# (https://github.com/alibaba/easyrobust/tree/main/benchmarks/coco_o), then:
python data/download_coco_o.py --zip path/to/coco_o.zip

# Objects365 (training) — streams patch0 tarball + HF annotations
pip install datasets
python data/download_objects365.py
```

Outputs land at the paths the training command below expects.

## Usage

```bash
python train.py --config configs/default.yaml \
    --train-img-dir data/objects365/train \
    --train-ann     data/objects365/annotations/train.json \
    --val-img-dir   data/coco/val2017 \
    --val-ann       data/coco/annotations/instances_val2017.json \
    --coco-o-img-dir data/coco-o/images \
    --coco-o-ann     data/coco-o/annotations/coco_o.json \
    --checkpoint    checkpoints/base/vit/model_best.pth.tar \
    --epochs 50 \
    --lr 1e-4 \
    --batch-size 4 \
    --eval-interval 5
```

## Project Structure

```
core/
  __init__.py            # public API
  backbone.py            # FrozenVitBackbone — frozen timm ViT
  detr_head.py           # DETRHead — transformer decoder + prediction heads
  det_model.py           # DetectionModel — backbone + head composition
  matcher.py             # HungarianMatcher — bipartite assignment
  losses.py              # DetectionLoss — CE + L1 + GIoU
  position_encoding.py   # PositionEncoding2D — fixed sin/cos grid encoding
  datasets.py            # CocoFormatDataset — works with Objects365, COCO, COCO-O
  class_mapping.py       # build_category_mapping — Objects365 ↔ COCO by name
  coco_eval.py           # evaluate_coco_map — pycocotools mAP evaluation
train.py                 # training entry point
evaluations/             # evaluation runners (single model + sweeps)
checkpoints/base/        # pretrained backbone weights
checkpoints/trained/     # trained head checkpoints + resume states
```

## Known Issues

### 1. Docstring / implementation mismatch in `core/losses.py`

The module docstring and `DetectionLoss` class docstring both reference "Focal classification loss," but `_loss_classification` actually uses `F.cross_entropy` with a weighted no-object class. The `sigmoid_focal_loss` function is defined at the top of the file but never called. The `focal_alpha` and `focal_gamma` constructor parameters are accepted but unused (dead code).

**Impact:** No functional bug — original DETR uses CE, not focal loss. The docstrings are misleading and the dead code adds confusion.

### 2. No data augmentation for detection

`CocoFormatDataset` only applies `Resize → ToTensor → Normalize`. Detection fine-tuning typically benefits from horizontal flip, color jitter, and scale jitter. Additionally, the current resize squashes images to a fixed square without preserving aspect ratio, which distorts object shapes and can hurt bbox regression quality. Aspect-ratio-preserving resize with padding is more standard for detection pipelines.

**Impact:** Likely to limit final detection accuracy, especially on objects with extreme aspect ratios.

### 3. Missing `pytorch-image-models` dependency

`train.py` imports timm-backed models. Training will fail unless either:
- The `pytorch-image-models` repo is cloned into the project root, or
- `timm` is installed via `pip install timm` and the `sys.path` hack is removed.

**Impact:** Script will not run out of the box.

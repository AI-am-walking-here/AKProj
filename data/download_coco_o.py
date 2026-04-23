"""Build a COCO-O subset from a user-provided zip.

You download the COCO-O zip manually (from the official source:
https://github.com/alibaba/easyrobust/tree/main/benchmarks/coco_o) and point
this script at it with --zip. The script extracts it, finds the annotation
JSON(s), and writes a flat subset.

Outputs:
    data/coco-o/images/*.jpg             (flattened; domain info preserved in JSON)
    data/coco-o/annotations/coco_o.json  (COCO-format JSON over the kept images)

Usage:
    python data/download_coco_o.py --zip C:/Users/me/Downloads/coco_o.zip
    python data/download_coco_o.py --zip coco_o.zip --num-images 5000 --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent
COCO_O_ROOT = DATA_ROOT / "coco-o"
RAW_DIR = COCO_O_ROOT / "_raw"
OUT_IMG_DIR = COCO_O_ROOT / "images"
OUT_ANN_FILE = COCO_O_ROOT / "annotations" / "coco_o.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SubsetConfig:
    zip_path: Path
    num_images: int
    force: bool


def _extract_zip(zip_path: Path, out_dir: Path, *, force: bool) -> None:
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        _logger.info(f"Already extracted at {out_dir}, reusing.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"Extracting {zip_path.name} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def _find_annotation_jsons(raw_root: Path) -> List[Path]:
    """COCO-O's zip may contain one unified JSON or per-domain JSONs.

    Prefer files whose name or parent dir looks like COCO annotations.
    """
    all_jsons = sorted(raw_root.rglob("*.json"))
    if not all_jsons:
        raise FileNotFoundError(f"No .json files found under {raw_root}")
    preferred = [
        p for p in all_jsons
        if "instance" in p.name.lower()
        or "coco" in p.name.lower()
        or "annotation" in p.parent.name.lower()
    ]
    return preferred or all_jsons


def _load_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _merge_coco(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple COCO-format dicts into one, re-keying ids to avoid collisions."""
    out_images: List[Dict] = []
    out_anns: List[Dict] = []
    cats_by_name: Dict[str, Dict] = {}
    next_img_id = 1
    next_ann_id = 1
    next_cat_id = 1

    for d in dicts:
        img_id_map: Dict[int, int] = {}
        cat_id_map: Dict[int, int] = {}

        for cat in d.get("categories", []):
            name = cat["name"]
            if name not in cats_by_name:
                cats_by_name[name] = {"id": next_cat_id, "name": name}
                next_cat_id += 1
            cat_id_map[cat["id"]] = cats_by_name[name]["id"]

        for im in d.get("images", []):
            new_id = next_img_id
            next_img_id += 1
            img_id_map[im["id"]] = new_id
            out_images.append({**im, "id": new_id, "file_name": os.path.basename(im["file_name"])})

        for ann in d.get("annotations", []):
            if ann.get("image_id") not in img_id_map:
                continue
            out_anns.append({
                **ann,
                "id": next_ann_id,
                "image_id": img_id_map[ann["image_id"]],
                "category_id": cat_id_map.get(ann.get("category_id"), ann.get("category_id")),
            })
            next_ann_id += 1

    return {
        "images": out_images,
        "annotations": out_anns,
        "categories": list(cats_by_name.values()),
    }


def _index_images(raw_root: Path) -> Dict[str, Path]:
    """Map basename -> full path for every image under raw_root (first match wins)."""
    idx: Dict[str, Path] = {}
    for p in raw_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            idx.setdefault(p.name, p)
    return idx


def _sample_images(images: List[Dict], n: int) -> List[Dict]:
    if len(images) <= n:
        return images
    return random.sample(images, n)


def _filter_anns(anns: List[Dict], keep_ids: Set[int]) -> List[Dict]:
    return [a for a in anns if a.get("image_id") in keep_ids]


def _copy_subset_images(images: List[Dict], idx: Dict[str, Path], dst: Path) -> Tuple[int, List[str]]:
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing: List[str] = []
    for im in images:
        fn = os.path.basename(im["file_name"])
        src = idx.get(fn)
        if src is None:
            missing.append(fn)
            continue
        shutil.copy2(src, dst / fn)
        copied += 1
    return copied, missing


def _write_json(p: Path, obj: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def build_subset(cfg: SubsetConfig) -> Tuple[Path, Path]:
    if not cfg.zip_path.exists():
        raise FileNotFoundError(f"--zip not found: {cfg.zip_path}")

    if cfg.force:
        for p in (OUT_IMG_DIR, OUT_ANN_FILE.parent):
            if p.exists():
                shutil.rmtree(p)

    _extract_zip(cfg.zip_path, RAW_DIR, force=cfg.force)

    ann_jsons = _find_annotation_jsons(RAW_DIR)
    _logger.info(f"Found {len(ann_jsons)} annotation file(s): {[p.name for p in ann_jsons]}")
    merged = _merge_coco(_load_json(p) for p in ann_jsons)
    _logger.info(
        f"Merged COCO: {len(merged['images'])} images, "
        f"{len(merged['annotations'])} anns, "
        f"{len(merged['categories'])} categories"
    )

    kept = _sample_images(merged["images"], cfg.num_images)
    keep_ids: Set[int] = {im["id"] for im in kept}
    out = {
        "images": kept,
        "annotations": _filter_anns(merged["annotations"], keep_ids),
        "categories": merged["categories"],
    }

    idx = _index_images(RAW_DIR)
    copied, missing = _copy_subset_images(kept, idx, OUT_IMG_DIR)
    _write_json(OUT_ANN_FILE, out)

    _logger.info(f"Copied {copied}/{len(kept)} images to {OUT_IMG_DIR}")
    if missing:
        _logger.warning(f"  {len(missing)} filenames not found in zip. Example: {missing[:5]}")
    _logger.info(f"Wrote {OUT_ANN_FILE}")
    return OUT_IMG_DIR, OUT_ANN_FILE


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a COCO-O subset from a manually-downloaded zip.")
    p.add_argument("--zip", type=Path, required=True, help="Path to the COCO-O zip you downloaded.")
    p.add_argument("--num-images", type=int, default=5000, help="Max images to keep (default 5000).")
    p.add_argument("--force", action="store_true", help="Re-extract zip and rebuild subset from scratch.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = SubsetConfig(
        zip_path=args.zip.resolve(),
        num_images=int(args.num_images),
        force=bool(args.force),
    )
    build_subset(cfg)


if __name__ == "__main__":
    main()

"""Download data for smoke-testing the detection pipeline.

Two modes:
  1. COCO val2017 (default) — downloads images + annotations directly.
     Small (~1 GB images, 0.25 MB annotations), works end-to-end.

  2. Objects365 annotations from HuggingFace — annotation-only.
     Images must be obtained separately from the Objects365 servers.

Usage:
    # Download COCO val2017 (recommended for smoke test)
    python download_data.py --dataset coco

    # Download Objects365 annotations only (1k samples from HF)
    python download_data.py --dataset objects365 --num-samples 1000

Requires: pip install datasets (for Objects365 HF download)
"""
import argparse
import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent / "data"

# ---- COCO val2017 -------------------------------------------------------- #

COCO_URLS = {
    "images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _robust_download(url: str, dest: Path, max_retries: int = 5) -> None:
    """Download a file with resume support and retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = dest.stat().st_size if dest.exists() else 0

    for attempt in range(1, max_retries + 1):
        try:
            headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
            resp = requests.get(url, headers=headers, stream=True, timeout=30)

            total = int(resp.headers.get("content-length", 0)) + downloaded
            mode = "ab" if downloaded > 0 else "wb"

            if downloaded > 0:
                _logger.info(f"  Resuming from {downloaded / 1e6:.1f} MB")

            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded * 100 // total if total > 0 else 0
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct}%)",
                          end="", flush=True)

            print()
            return

        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            _logger.warning(f"  Download interrupted (attempt {attempt}/{max_retries}): {e}")
            downloaded = dest.stat().st_size if dest.exists() else 0
            if attempt < max_retries:
                wait = 5 * attempt
                _logger.info(f"  Retrying in {wait}s ...")
                time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def download_coco(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    img_dir = dest / "val2017"
    ann_file = dest / "annotations" / "instances_val2017.json"

    if img_dir.exists() and any(img_dir.iterdir()):
        _logger.info(f"COCO images already present at {img_dir}, skipping")
    else:
        zip_path = dest / "val2017.zip"
        _logger.info("Downloading COCO val2017 images (~800 MB) ...")
        _robust_download(COCO_URLS["images"], zip_path)
        _logger.info("Extracting images ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()

    if ann_file.exists():
        _logger.info(f"COCO annotations already present at {ann_file}, skipping")
    else:
        zip_path = dest / "annotations.zip"
        _logger.info("Downloading COCO annotations ...")
        _robust_download(COCO_URLS["annotations"], zip_path)
        _logger.info("Extracting annotations ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()

    _logger.info(f"COCO val2017 ready at {dest}")
    _logger.info(f"  Images:      {img_dir}")
    _logger.info(f"  Annotations: {ann_file}")


# ---- Objects365 from HuggingFace ----------------------------------------- #

def download_objects365_annotations(dest: Path, num_samples: int = 1000) -> None:
    """Stream Objects365 annotations from HF and build COCO-format JSON.

    NOTE: This downloads annotations only. Images must be obtained
    separately from the Objects365 servers.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets  (needed for HF streaming)")

    dest.mkdir(parents=True, exist_ok=True)
    ann_dir = dest / "annotations"
    ann_dir.mkdir(exist_ok=True)
    img_dir = dest / "train"
    img_dir.mkdir(exist_ok=True)

    ann_file = ann_dir / "train.json"
    if ann_file.exists():
        _logger.info(f"Objects365 annotations already present at {ann_file}, skipping")
        return

    _logger.info(f"Streaming {num_samples} samples from jxu124/objects365 ...")
    ds = load_dataset("jxu124/objects365", split="train", streaming=True)
    samples = list(ds.take(num_samples))
    _logger.info(f"Downloaded {len(samples)} annotation records")

    categories_seen: Dict[int, str] = {}
    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []
    ann_id_counter = 1

    for sample in samples:
        img_info = sample["image_info"]
        coco_images.append({
            "id": img_info["id"],
            "file_name": os.path.basename(img_info["file_name"]),
            "width": img_info["width"],
            "height": img_info["height"],
        })

        for ann in sample["anns_info"]:
            cat_id = ann["category_id"]
            if cat_id not in categories_seen:
                categories_seen[cat_id] = ann["category"]

            x1, y1, x2, y2 = ann["bbox"]
            w = x2 - x1
            h = y2 - y1

            coco_annotations.append({
                "id": ann_id_counter,
                "image_id": ann["image_id"],
                "category_id": cat_id,
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "area": round(ann["area"], 2),
                "iscrowd": ann["iscrowd"],
            })
            ann_id_counter += 1

    coco_categories = [
        {"id": cat_id, "name": name}
        for cat_id, name in sorted(categories_seen.items())
    ]

    coco_json = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    with open(ann_file, "w") as f:
        json.dump(coco_json, f)

    _logger.info(f"Objects365 annotations saved to {ann_file}")
    _logger.info(f"  {len(coco_images)} images, {len(coco_annotations)} annotations, "
                 f"{len(coco_categories)} categories")
    _logger.info(f"  NOTE: Images must be placed in {img_dir}/ manually")
    _logger.info(f"  Expected filenames: {coco_images[0]['file_name']}, ...")


# ---- CLI ----------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Download data for detection smoke test")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["coco", "objects365"],
                        help="Which dataset to download")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of Objects365 samples to stream (default: 1000)")
    args = parser.parse_args()

    if args.dataset == "coco":
        download_coco(DATA_ROOT / "coco")
    elif args.dataset == "objects365":
        download_objects365_annotations(DATA_ROOT / "objects365", args.num_samples)


if __name__ == "__main__":
    main()

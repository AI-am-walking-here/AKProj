"""Build an Objects365 training subset from patch0 + HF annotations.

Strategy (single-patch, chosen for tractability):
    1. Stream the Objects365 train patch0 tarball from KS3; extract the
       first N images we encounter and stop. No need to fully download
       ~15 GB; we close the connection as soon as we have enough.
    2. Stream `jxu124/objects365` on Hugging Face (annotations only) and
       collect COCO-format annotations for exactly the images we kept.
    3. Write a clean COCO-format JSON.

Outputs:
    data/objects365/train/*.jpg
    data/objects365/annotations/train.json

Usage:
    python data/download_objects365.py
    python data/download_objects365.py --num-images 5000 --force
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent
OBJ_ROOT = DATA_ROOT / "objects365"
IMG_DIR = OBJ_ROOT / "train"
ANN_FILE = OBJ_ROOT / "annotations" / "train.json"

PATCH0_URL = (
    "https://dorc.ks3-cn-beijing.ksyun.com/data-set/"
    "2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/patch0.tar.gz"
)
HF_DATASET = "jxu124/objects365"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class FetchConfig:
    num_images: int
    force: bool
    connect_timeout_s: int = 20
    read_timeout_s: int = 60
    max_retries: int = 5
    backoff_s: int = 5
    progress_every_s: float = 10.0


def _timeout(cfg: FetchConfig) -> Tuple[int, int]:
    return (cfg.connect_timeout_s, cfg.read_timeout_s)


def _existing_images(dst: Path) -> Set[str]:
    if not dst.exists():
        return set()
    return {p.name for p in dst.glob("*") if p.suffix.lower() in IMAGE_EXTS}


def _stream_extract_first_n(url: str, dst: Path, n: int, cfg: FetchConfig) -> Set[str]:
    """Stream a .tar.gz from URL and extract up to `n` images into `dst`.

    Returns the set of basenames written (including pre-existing ones).
    """
    dst.mkdir(parents=True, exist_ok=True)
    kept: Set[str] = _existing_images(dst)
    if len(kept) >= n:
        _logger.info(f"Already have {len(kept)} images; skipping tarball download.")
        return set(list(kept)[:n])

    for attempt in range(1, cfg.max_retries + 1):
        started = time.monotonic()
        last_log = started
        bytes_in = 0
        try:
            with requests.get(url, stream=True, timeout=_timeout(cfg)) as resp:
                resp.raise_for_status()
                resp.raw.decode_content = True

                class _Counting(io.RawIOBase):
                    def readable(self) -> bool:
                        return True

                    def readinto(self, b) -> int:  # type: ignore[override]
                        nonlocal bytes_in, last_log
                        read = resp.raw.readinto(b)
                        if read:
                            bytes_in += int(read)
                        now = time.monotonic()
                        if now - last_log >= cfg.progress_every_s:
                            mb = bytes_in / 1e6
                            elapsed = max(1e-3, now - started)
                            _logger.info(
                                f"  streamed={mb:.1f} MB @ {mb/elapsed:.2f} MB/s | "
                                f"kept={len(kept)}/{n}"
                            )
                            last_log = now
                        return int(read or 0)

                    def read(self, size: int = -1) -> bytes:
                        nonlocal bytes_in
                        chunk = resp.raw.read(size)
                        if chunk:
                            bytes_in += len(chunk)
                        return chunk

                raw = io.BufferedReader(_Counting())
                with tarfile.open(fileobj=raw, mode="r|gz") as tf:
                    for member in tf:
                        if not member.isfile():
                            continue
                        name = os.path.basename(member.name)
                        if Path(name).suffix.lower() not in IMAGE_EXTS:
                            continue
                        out_path = dst / name
                        if out_path.exists():
                            kept.add(name)
                        else:
                            f = tf.extractfile(member)
                            if f is None:
                                continue
                            with open(out_path, "wb") as w:
                                w.write(f.read())
                            kept.add(name)

                        if len(kept) >= n:
                            _logger.info(f"Reached {n} images; closing stream.")
                            return set(list(kept)[:n])

            if len(kept) >= 1:
                _logger.warning(
                    f"Stream ended with only {len(kept)}/{n} images extracted."
                )
                return set(list(kept)[:n])

        except (requests.RequestException, tarfile.TarError, OSError) as e:
            _logger.warning(f"  Stream/extract failed (attempt {attempt}/{cfg.max_retries}): {e}")
            if attempt < cfg.max_retries:
                wait = cfg.backoff_s * attempt
                _logger.info(f"  Retrying in {wait}s ...")
                time.sleep(wait)

    raise RuntimeError(f"Failed to stream {url} after {cfg.max_retries} attempts")


def _collect_annotations(needed: Set[str]) -> Dict:
    """Stream HF annotations and collect entries for filenames in `needed`."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets  (required for HF annotations)") from e

    _logger.info(
        f"Streaming {HF_DATASET} to collect annotations for {len(needed)} images ..."
    )
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []
    categories_seen: Dict[int, str] = {}
    matched: Set[str] = set()
    ann_id = 1
    scanned = 0
    last_log = time.monotonic()

    for sample in ds:
        scanned += 1
        img_info = sample.get("image_info") or {}
        fn = os.path.basename(img_info.get("file_name") or "")
        if fn and fn in needed and fn not in matched:
            matched.add(fn)
            coco_images.append({
                "id": int(img_info["id"]),
                "file_name": fn,
                "width": int(img_info["width"]),
                "height": int(img_info["height"]),
            })
            for ann in sample.get("anns_info") or []:
                cid = int(ann["category_id"])
                categories_seen.setdefault(cid, ann["category"])
                x1, y1, x2, y2 = ann["bbox"]
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": int(ann["image_id"]),
                    "category_id": cid,
                    "bbox": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(x2 - x1), 2),
                        round(float(y2 - y1), 2),
                    ],
                    "area": round(float(ann["area"]), 2),
                    "iscrowd": int(ann["iscrowd"]),
                })
                ann_id += 1

        now = time.monotonic()
        if now - last_log >= 10.0:
            _logger.info(f"  scanned={scanned:,} | matched={len(matched)}/{len(needed)}")
            last_log = now

        if len(matched) >= len(needed):
            break

    if len(matched) < len(needed):
        _logger.warning(
            f"Only matched {len(matched)}/{len(needed)} images in HF stream."
        )

    coco_categories = [{"id": cid, "name": name} for cid, name in sorted(categories_seen.items())]
    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }


def _write_json(p: Path, obj: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _prune_orphan_images(kept: Set[str], annotated: Set[str], img_dir: Path) -> int:
    """Delete images we kept but couldn't find annotations for (keeps JSON + disk in sync)."""
    orphans = kept - annotated
    for name in orphans:
        p = img_dir / name
        if p.exists():
            p.unlink()
    return len(orphans)


def fetch(cfg: FetchConfig) -> Tuple[Path, Path]:
    OBJ_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ANN_FILE.parent.mkdir(parents=True, exist_ok=True)

    if cfg.force:
        _logger.info("Force enabled: wiping existing train/ and annotations/.")
        for p in IMG_DIR.glob("*"):
            if p.is_file():
                p.unlink()
        if ANN_FILE.exists():
            ANN_FILE.unlink()

    _logger.info(f"Extracting up to {cfg.num_images} images from patch0 (streamed) ...")
    kept = _stream_extract_first_n(PATCH0_URL, IMG_DIR, cfg.num_images, cfg)
    _logger.info(f"Images on disk: {len(kept)}")

    coco = _collect_annotations(kept)
    annotated_names = {im["file_name"] for im in coco["images"]}
    dropped = _prune_orphan_images(kept, annotated_names, IMG_DIR)
    if dropped:
        _logger.warning(f"Dropped {dropped} images with no matching HF annotations.")

    _write_json(ANN_FILE, coco)
    _logger.info(
        f"Done. {len(coco['images'])} images, {len(coco['annotations'])} anns, "
        f"{len(coco['categories'])} categories."
    )
    _logger.info(f"      Images: {IMG_DIR}")
    _logger.info(f"      Anns:   {ANN_FILE}")
    return IMG_DIR, ANN_FILE


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an Objects365 subset from patch0 + HF annotations.")
    p.add_argument("--num-images", type=int, default=5000, help="Target number of training images (default 5000).")
    p.add_argument("--force", action="store_true", help="Wipe existing images + annotations before rebuilding.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    fetch(FetchConfig(num_images=int(args.num_images), force=bool(args.force)))


if __name__ == "__main__":
    main()

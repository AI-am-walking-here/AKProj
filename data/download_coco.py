"""Download COCO val2017 (images + annotations).

val2017 is exactly 5000 images, so no subsetting is required.

Outputs:
    data/coco/val2017/*.jpg
    data/coco/annotations/instances_val2017.json

Usage:
    python data/download_coco.py
    python data/download_coco.py --force        # re-download even if present
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent
COCO_ROOT = DATA_ROOT / "coco"
IMG_DIR = COCO_ROOT / "val2017"
ANN_FILE = COCO_ROOT / "annotations" / "instances_val2017.json"
DL_DIR = COCO_ROOT / "_downloads"

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


@dataclass(frozen=True)
class DownloadConfig:
    chunk_size: int = 1 << 20            # 1 MiB
    timeout_s: int = 30
    max_retries: int = 5
    backoff_s: int = 5
    progress_every_s: float = 2.0


def _download(url: str, dest: Path, cfg: DownloadConfig) -> None:
    """Stream a URL to disk with resume + retry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    for attempt in range(1, cfg.max_retries + 1):
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            resp = requests.get(url, headers=headers, stream=True, timeout=cfg.timeout_s)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) + have
            mode = "ab" if have else "wb"
            if have:
                _logger.info(f"  Resuming {dest.name} from {have/1e6:.1f} MB")

            started = time.monotonic()
            last = started
            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=cfg.chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    have += len(chunk)
                    now = time.monotonic()
                    if now - last >= cfg.progress_every_s:
                        _log_progress(have, total, started, now)
                        last = now
            return

        except requests.RequestException as e:
            _logger.warning(f"  Download failed (attempt {attempt}/{cfg.max_retries}): {e}")
            have = dest.stat().st_size if dest.exists() else 0
            if attempt < cfg.max_retries:
                time.sleep(cfg.backoff_s * attempt)

    raise RuntimeError(f"Failed to download {url}")


def _log_progress(have: int, total: int, started: float, now: float) -> None:
    mb = have / 1e6
    elapsed = max(1e-3, now - started)
    if total > 0:
        _logger.info(f"  {mb:.1f} MB ({have*100/total:.1f}%) @ {mb/elapsed:.2f} MB/s")
    else:
        _logger.info(f"  {mb:.1f} MB @ {mb/elapsed:.2f} MB/s")


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def _already_have_images() -> bool:
    return IMG_DIR.exists() and any(IMG_DIR.glob("*.jpg"))


def _already_have_annotations() -> bool:
    return ANN_FILE.exists() and ANN_FILE.stat().st_size > 0


def fetch(*, force: bool, cfg: DownloadConfig) -> None:
    COCO_ROOT.mkdir(parents=True, exist_ok=True)
    DL_DIR.mkdir(parents=True, exist_ok=True)

    if force:
        for p in (IMG_DIR, ANN_FILE.parent, DL_DIR):
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        DL_DIR.mkdir(parents=True, exist_ok=True)

    if _already_have_images():
        _logger.info(f"Images already present at {IMG_DIR}, skipping.")
    else:
        img_zip = DL_DIR / "val2017.zip"
        _logger.info("Downloading val2017 images (~1 GB) ...")
        _download(IMAGES_URL, img_zip, cfg)
        _logger.info("Extracting images ...")
        _extract_zip(img_zip, COCO_ROOT)

    if _already_have_annotations():
        _logger.info(f"Annotations already present at {ANN_FILE}, skipping.")
    else:
        ann_zip = DL_DIR / "annotations_trainval2017.zip"
        _logger.info("Downloading annotations ...")
        _download(ANNOTATIONS_URL, ann_zip, cfg)
        _logger.info("Extracting annotations ...")
        _extract_zip(ann_zip, COCO_ROOT)

    _logger.info(f"Done. Images: {IMG_DIR}")
    _logger.info(f"      Anns:   {ANN_FILE}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download COCO val2017 (5000 images + annotations).")
    p.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fetch(force=bool(args.force), cfg=DownloadConfig())


if __name__ == "__main__":
    main()

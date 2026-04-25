"""Build a COCO train2017 subset (N images + filtered annotations).

Rationale
    The full COCO train2017 set is ~18 GB of images. For laptop-scale DINOv3-style
    fine-tuning we want a faithful slice, not the whole pie. This script:

        1. Reads the existing `instances_train2017.json` (already shipped by
           `download_coco.py` via the annotations zip).
        2. Deterministically samples N image IDs (seeded, reproducible).
        3. Downloads just those images individually from the public COCO
           bucket (`http://images.cocodataset.org/train2017/<file>`).
        4. Writes a filtered COCO-format JSON containing only those images
           and their annotations.

Outputs:
    data/coco/train2017_subset/*.jpg
    data/coco/annotations/instances_train2017_subset.json

Usage:
    python data/download_coco_train.py                      # default: 5000 images, seed 42
    python data/download_coco_train.py --num-images 2000
    python data/download_coco_train.py --force              # re-download missing
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent
COCO_ROOT = DATA_ROOT / "coco"
FULL_ANN = COCO_ROOT / "annotations" / "instances_train2017.json"
SUBSET_IMG_DIR = COCO_ROOT / "train2017_subset"
SUBSET_ANN = COCO_ROOT / "annotations" / "instances_train2017_subset.json"

IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/train2017/{name}"


@dataclass(frozen=True)
class Config:
    num_images: int = 5000
    seed: int = 42
    timeout_s: int = 30
    max_retries: int = 4
    backoff_s: int = 3
    log_every: int = 100
    workers: int = 16             # parallel downloads; COCO's bucket handles this well


# --------------------------------------------------------------------------- #
# COCO JSON filtering (pure function; no I/O)
# --------------------------------------------------------------------------- #

def _filter_coco(full: Dict, keep_image_ids: set) -> Dict:
    """Return a new COCO dict containing only the selected images and their anns."""
    images = [im for im in full["images"] if im["id"] in keep_image_ids]
    anns = [a for a in full["annotations"] if a["image_id"] in keep_image_ids]
    return {
        "info": full.get("info", {}),
        "licenses": full.get("licenses", []),
        "categories": full["categories"],
        "images": images,
        "annotations": anns,
    }


def _pick_image_ids(full: Dict, n: int, seed: int) -> List[int]:
    """Deterministic reproducible sampling of image ids."""
    ids = sorted(im["id"] for im in full["images"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:n]


# --------------------------------------------------------------------------- #
# Per-image download with retry (isolated for testability)
# --------------------------------------------------------------------------- #

def _download_one(url: str, dest: Path, cfg: Config, session: requests.Session) -> bool:
    """Download a single image. Returns True on success, False on give-up.

    Uses a shared `Session` so the connection pool and TLS handshakes are
    reused across requests on the same thread (and across threads, since
    `requests.Session` is thread-safe for these simple GETs).
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, cfg.max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=cfg.timeout_s) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            # Windows antivirus/indexer can briefly hold the freshly-written file,
            # causing `os.replace` to raise WinError 32. Short retry loop masks it.
            for rename_try in range(5):
                try:
                    tmp.replace(dest)
                    return True
                except PermissionError:
                    time.sleep(0.2 * (rename_try + 1))
            # Give up cleanly after 5 retries; treat as a normal download failure.
            _logger.debug(f"    {dest.name}: rename locked by OS after 5 retries")
            return False
        except requests.RequestException as e:
            _logger.debug(f"    {dest.name}: attempt {attempt} failed ({e})")
            if attempt < cfg.max_retries:
                time.sleep(cfg.backoff_s * attempt)
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return False


def _download_many(
    tasks: List[Dict],           # list of {"url": ..., "dest": Path}
    cfg: Config,
) -> Dict[str, int]:
    """Drive parallel downloads. Returns {'ok': int, 'fail': int}."""
    session = requests.Session()
    # Bump connection pool to match worker count so threads don't serialize.
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=cfg.workers, pool_maxsize=cfg.workers
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    ok = fail = 0
    log_lock = Lock()
    total = len(tasks)

    def _run(task: Dict) -> bool:
        return _download_one(task["url"], task["dest"], cfg, session)

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = [pool.submit(_run, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            else:
                fail += 1
            if i % cfg.log_every == 0:
                with log_lock:
                    _logger.info(f"  [{i}/{total}] ok={ok} fail={fail}")
    return {"ok": ok, "fail": fail}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_subset(*, force: bool, cfg: Config) -> None:
    if not FULL_ANN.exists():
        raise FileNotFoundError(
            f"Missing {FULL_ANN}. Run `python data/download_coco.py` first — "
            f"the annotations zip it pulls includes instances_train2017.json."
        )

    SUBSET_IMG_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_ANN.parent.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Loading full train2017 annotations from {FULL_ANN} ...")
    with open(FULL_ANN, "r", encoding="utf-8") as f:
        full = json.load(f)
    _logger.info(f"  {len(full['images']):,} images, {len(full['annotations']):,} annotations")

    keep_ids = set(_pick_image_ids(full, cfg.num_images, cfg.seed))
    subset_images = [im for im in full["images"] if im["id"] in keep_ids]
    _logger.info(f"Sampled {len(keep_ids)} image ids (seed={cfg.seed}).")

    # Build work list; optionally purge existing files when --force.
    tasks: List[Dict] = []
    for im in subset_images:
        dest = SUBSET_IMG_DIR / im["file_name"]
        if force and dest.exists():
            dest.unlink()
        tasks.append({"url": IMAGE_URL_TEMPLATE.format(name=im["file_name"]), "dest": dest})

    _logger.info(f"Downloading {len(tasks)} images with {cfg.workers} workers ...")
    counts = _download_many(tasks, cfg)
    _logger.info(f"Downloads: ok={counts['ok']}  fail={counts['fail']}")

    # Keep only successfully-downloaded images in the output JSON. Missing images
    # would break the data loader silently, so we filter proactively.
    present_names = {p.name for p in SUBSET_IMG_DIR.iterdir() if p.is_file()}
    final_ids = {im["id"] for im in subset_images if im["file_name"] in present_names}
    subset = _filter_coco(full, final_ids)
    _logger.info(
        f"Writing subset JSON: {len(subset['images'])} images, "
        f"{len(subset['annotations'])} annotations -> {SUBSET_ANN}"
    )
    with open(SUBSET_ANN, "w", encoding="utf-8") as f:
        json.dump(subset, f)

    _logger.info("Done.")
    _logger.info(f"  Images: {SUBSET_IMG_DIR}")
    _logger.info(f"  Anns:   {SUBSET_ANN}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a COCO train2017 subset (images + filtered annotations).")
    p.add_argument("--num-images", type=int, default=5000, help="Number of training images to sample (default: 5000).")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed for reproducibility (default: 42).")
    p.add_argument("--workers", type=int, default=16, help="Concurrent download threads (default: 16).")
    p.add_argument("--force", action="store_true", help="Re-download even if a file already exists on disk.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_subset(
        force=bool(args.force),
        cfg=Config(
            num_images=int(args.num_images),
            seed=int(args.seed),
            workers=max(1, int(args.workers)),
        ),
    )


if __name__ == "__main__":
    main()

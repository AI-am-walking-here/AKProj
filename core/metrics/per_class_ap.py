"""Per-category AP breakdown using pycocotools.

Returns a dict of {category_name: AP} for every category present in the
ground-truth annotation file, enabling fine-grained analysis beyond the
standard 12 aggregate COCO metrics.
"""
import logging
from typing import Dict, List

import numpy as np

_logger = logging.getLogger(__name__)


def compute_per_class_ap(
    results: List[Dict],
    ann_file: str,
    iou_type: str = "bbox",
) -> Dict[str, float]:
    """Compute per-category AP@[0.5:0.95] for every category in *ann_file*.

    Args:
        results: List of ``{"image_id", "category_id", "bbox", "score"}``
            dicts produced by ``predictions_to_coco_results``.
        ann_file: Path to the COCO ground-truth annotation JSON.
        iou_type: IoU type for evaluation — ``"bbox"`` (default) or
            ``"segm"``.

    Returns:
        ``{category_name: AP_float}`` where AP is in [0, 1].
        Categories with no ground-truth or no predictions have AP = 0.0.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)

    if not results:
        _logger.warning("No predictions — all per-class APs are 0.0")
        cat_names = [c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())]
        return {n: 0.0 for n in cat_names}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    # coco_eval.eval['precision'] shape:
    #   (num_iou_thresholds, num_recall_thresholds, num_categories, num_areas, num_max_dets)
    # We want AP@[0.5:0.95] = mean over IoU thresholds, area=all (index 0), maxDets=100 (index 2)
    precision = coco_eval.eval["precision"]  # (T, R, K, A, M)

    cat_ids = coco_gt.getCatIds()
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    # Build a mapping from category ID → index in the COCOeval internal array.
    # COCOeval orders categories by the sorted list of IDs it was evaluated on.
    eval_cat_ids = coco_eval.params.catIds
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(eval_cat_ids)}

    per_class_ap: Dict[str, float] = {}
    for cat_id in cat_ids:
        name = cat_id_to_name[cat_id]
        if cat_id not in cat_id_to_idx:
            per_class_ap[name] = 0.0
            continue

        idx = cat_id_to_idx[cat_id]
        # precision[:, :, idx, 0, 2] → all IoU thresholds, all recall points,
        # area=all, maxDets=100
        cat_precision = precision[:, :, idx, 0, 2]
        valid = cat_precision[cat_precision >= 0]
        per_class_ap[name] = float(valid.mean()) if valid.size > 0 else 0.0

    return per_class_ap

"""Standard 12-metric COCO evaluation using pycocotools.

Returns AP, AP50, AP75, AP_small, AP_medium, AP_large,
AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large.
"""
import logging
from typing import Dict, List

_logger = logging.getLogger(__name__)

_METRIC_NAMES = [
    "AP", "AP50", "AP75",
    "AP_small", "AP_medium", "AP_large",
    "AR@1", "AR@10", "AR@100",
    "AR_small", "AR_medium", "AR_large",
]


def run_coco_evaluation(
    results: List[Dict],
    ann_file: str,
    iou_type: str = "bbox",
) -> Dict[str, float]:
    """Evaluate a list of COCO-format result dicts against ground truth.

    Args:
        results: List of ``{"image_id", "category_id", "bbox", "score"}``
            dicts produced by ``predictions_to_coco_results``.
        ann_file: Path to the COCO ground-truth annotation JSON.
        iou_type: IoU type for evaluation — ``"bbox"`` (default) or
            ``"segm"``.

    Returns:
        Dict with the 12 standard COCO metric keys:
        AP, AP50, AP75, AP_small, AP_medium, AP_large,
        AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large.
        All values are in [0, 1] (not multiplied by 100).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)

    if not results:
        _logger.warning("No predictions — returning zero metrics")
        return {n: 0.0 for n in _METRIC_NAMES}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {n: float(v) for n, v in zip(_METRIC_NAMES, coco_eval.stats)}

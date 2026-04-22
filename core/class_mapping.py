"""Category mapping between COCO-format datasets.

Builds a label-index mapping from a *source* dataset (e.g. Objects365) to a
*target* dataset (e.g. COCO) using case-insensitive name matching with an
overridable alias table for known mismatches.

Typical flow
-------------
1. ``build_category_mapping(obj365_ann, coco_ann)``
   → ``source_to_target_label``, ``target_label_to_cat_id``, unmatched names
2. Pass both dicts into ``coco_eval.evaluate_coco_map(...)`` at eval time.
3. Optionally persist / reload with ``save_mapping`` / ``load_mapping``.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# Objects365 name (lower) → COCO name (lower).
# ``None`` = known no-match (skip silently).
# Only entries whose names genuinely differ need to be here;
# exact-name matches are handled automatically.
_NAME_ALIASES: Dict[str, Optional[str]] = {
    "handbag/satchel": "handbag",
    "stuffed toy": "teddy bear",
    "wild bird": "bird",
    "cellphone": "cell phone",
    "television / monitor": "tv",
    "moniter / tv": "tv",
    "street lights": "traffic light",
    "sneakers": None,
    "other shoes": None,
    "leather shoes": None,
    "boots": None,
    "hat": None,
    "flower": None,
    "desk": None,
    "cabinet/shelf": None,
    "glasses": None,
    "belt": None,
    "sports car": None,
    "lamp": None,
    "storage box": None,
    "pen/pencil": None,
}


def load_categories(ann_file: str) -> List[Dict]:
    """Return the ``categories`` list from a COCO-format annotation file."""
    with open(ann_file, "r") as f:
        return json.load(f)["categories"]


def _normalise(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def build_category_mapping(
    source_ann_file: str,
    target_ann_file: str,
    aliases: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Dict[int, int], Dict[int, int], List[str]]:
    """Build a label-index mapping from *source* → *target* by category name.

    Args:
        source_ann_file: Annotation JSON for the source dataset
            (e.g. Objects365 training annotations).
        target_ann_file: Annotation JSON for the target dataset
            (e.g. COCO val annotations).
        aliases: Optional override for ``_NAME_ALIASES``.

    Returns:
        source_to_target_label:
            ``{source_label_idx: target_label_idx}`` for every matched pair.
        target_label_to_cat_id:
            ``{target_label_idx: target_category_id}`` — needed to convert
            predictions back to original COCO category IDs for pycocotools.
        unmatched:
            Source category names that could not be matched.
    """
    if aliases is None:
        aliases = _NAME_ALIASES

    source_cats = load_categories(source_ann_file)
    target_cats = load_categories(target_ann_file)

    target_name_to_label = {
        _normalise(cat["name"]): i for i, cat in enumerate(target_cats)
    }
    target_label_to_cat_id = {i: cat["id"] for i, cat in enumerate(target_cats)}

    source_to_target_label: Dict[int, int] = {}
    unmatched: List[str] = []

    for src_idx, cat in enumerate(source_cats):
        src_name = _normalise(cat["name"])

        if src_name in aliases:
            alias = aliases[src_name]
            if alias is not None and alias in target_name_to_label:
                source_to_target_label[src_idx] = target_name_to_label[alias]
            else:
                unmatched.append(cat["name"])
            continue

        if src_name in target_name_to_label:
            source_to_target_label[src_idx] = target_name_to_label[src_name]
        else:
            unmatched.append(cat["name"])

    _logger.info(
        f"Category mapping: {len(source_to_target_label)}/{len(source_cats)} "
        f"source categories matched, {len(unmatched)} unmatched"
    )
    return source_to_target_label, target_label_to_cat_id, unmatched


# ---- persistence helpers ----

def save_mapping(mapping: Dict[int, int], path: str) -> None:
    """Persist a label mapping to JSON (keys become strings)."""
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)


def load_mapping(path: str) -> Dict[int, int]:
    """Load a previously saved label mapping from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

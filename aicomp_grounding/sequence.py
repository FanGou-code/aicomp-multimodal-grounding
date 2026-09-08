"""Sequence-level fingerprints and annotation-product text QC."""

from __future__ import annotations

import re

from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.query import clean_query_text, validate_generated_query

_ANNOTATION_SCAFFOLD = (
    "highlighted",
    "marked target",
    "marked object",
    "outlined target",
    "red rectangle",
    "red outline",
    "annotated image",
    "annotation view",
    "this frame",
    "current frame",
    "in the frame",
    "within the frame",
    "this image",
    "current image",
    "in the image",
    "within the image",
    "crop",
    "same target",
    "visible image",
    "thermal image",
    "infrared image",
    "depth image",
    "depth map",
)
# NOTE: "image" is deliberately absent here. The official test dialect uses
# "of the image" as a frame anchor (303 occurrences) and v5 targets that
# dialect, so a blanket ban would reject legitimate queries. Real scaffolding
# ("this image", "in the image", "annotated image", ...) is still caught by
# _ANNOTATION_SCAFFOLD above.
_ANNOTATION_TERM = re.compile(
    r"\b(?:target|crop|annotation|annotated|bbox|coordinate|"
    r"infrared|thermal|depth|rgb)\b",
    flags=re.IGNORECASE,
)
_GENERIC_CATEGORY = re.compile(
    r"\b(?:thing|item|entity)\b",
    flags=re.IGNORECASE,
)


def source_fingerprint(dataset: dict) -> str:
    """Fingerprint immutable annotation inputs while deliberately ignoring Query."""
    identity = {}
    for key in sorted(dataset):
        item = dataset[key]
        identity[key] = {
            field: item.get(field)
            for field in ("visible", "infrared", "depth", "bbox", "width", "height")
        }
    return stable_json_hash(identity)


def validate_annotation_query(query: str) -> tuple[bool, str]:
    valid, reason = validate_generated_query(query, min_words=1, max_words=55)
    if not valid:
        return valid, reason
    lowered = clean_query_text(query).lower()
    marker = next(
        (
            phrase
            for phrase in _ANNOTATION_SCAFFOLD
            if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", lowered)
        ),
        None,
    )
    if marker:
        return False, f"query mentions annotation scaffolding {marker!r}"
    marker_match = _ANNOTATION_TERM.search(clean_query_text(query))
    if marker_match:
        return False, f"query mentions annotation term {marker_match.group(0)!r}"
    generic_match = _GENERIC_CATEGORY.search(clean_query_text(query))
    if generic_match:
        return False, f"query uses generic category {generic_match.group(0)!r}"
    return True, ""

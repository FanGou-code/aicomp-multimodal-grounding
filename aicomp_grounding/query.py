"""Validation for synthetic English visual-grounding queries.

Two layers: ``validate_generated_query`` checks format only; the
``validate_annotation_query`` wrapper adds the annotation-product rules (no
scaffolding vocabulary, no generic categories) that both the annotation
pipeline and the training-side contract enforce.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
_COORDINATES = re.compile(r"[\[(]\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?")
_CONTROL_TOKEN = re.compile(r"<\|[^<>\r\n]*\|>")
_FORBIDDEN = ("red rectangle", "red outline", "bounding box", "image 1", "image 2", "first image", "second image")

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
# "image" alone is not included: phrases such as "of the image" serve as legitimate
# spatial anchors; scaffolding phrases are matched via _ANNOTATION_SCAFFOLD.
_ANNOTATION_TERM = re.compile(
    r"\b(?:target|crop|annotation|annotated|bbox|coordinate|"
    r"infrared|thermal|depth|rgb)\b",
    flags=re.IGNORECASE,
)
_GENERIC_CATEGORY = re.compile(
    r"\b(?:thing|item|entity)\b",
    flags=re.IGNORECASE,
)


def clean_query_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip().strip("`").strip().strip('"').strip("'").strip()
    return cleaned.rstrip(".?!;").strip()


def validate_generated_query(text: str, *, min_words: int = 1, max_words: int = 55) -> tuple[bool, str]:
    """Validate format and obvious annotation failures without judging semantics."""
    cleaned = clean_query_text(text)
    if not cleaned:
        return False, "empty response"
    if "\n" in cleaned or "\r" in cleaned:
        return False, "query must be one line"
    if any(marker in cleaned.lower() for marker in _FORBIDDEN):
        return False, "query mentions annotation scaffolding"
    if _COORDINATES.search(cleaned):
        return False, "query contains coordinate-like output"
    if _CONTROL_TOKEN.search(cleaned):
        return False, "query contains a model control token"
    if cleaned.startswith(("#", "- ", "* ")):
        return False, "query contains markdown formatting"

    words = _WORD.findall(cleaned)
    if not min_words <= len(words) <= max_words:
        return False, f"expected {min_words}-{max_words} English words, got {len(words)}"
    if sum(character.isascii() for character in cleaned) / len(cleaned) < 0.9:
        return False, "query is not predominantly English/ASCII"
    return True, ""


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


def preflight_check_dataset(
    data: dict,
    *,
    split_name: str = "dataset",
    allow_pending: bool = False,
) -> list[str]:
    """Validate a dataset dict structurally before training.

    Returns a list of error strings; empty list means all checks passed.
    Only rejects structural problems (missing fields, empty queries,
    invalid bboxes) — does NOT enforce word-count or style rules.
    """
    from aicomp_grounding.bbox import validate_bbox

    errors: list[str] = []
    if not isinstance(data, dict) or not data:
        return [f"{split_name}: dataset is empty or is not a JSON object."]

    empty_ids: list[str] = []
    bad_bbox_ids: list[str] = []
    missing_field_ids: list[str] = []
    annotated_count = 0

    required_fields = ("visible", "infrared", "depth", "query", "bbox", "width", "height")

    for key, item in data.items():
        if not isinstance(item, dict):
            errors.append(f"{split_name}: sample {key!r} is not a dictionary.")
            continue
        for field in required_fields:
            if field not in item:
                missing_field_ids.append(key)
                break
        else:
            query = item.get("query", "")
            bbox = item.get("bbox")
            if not isinstance(query, str) or not query.strip():
                if not allow_pending:
                    empty_ids.append(key)
            else:
                annotated_count += 1
                if validate_bbox(bbox) is None:
                    bad_bbox_ids.append(key)

    if allow_pending and annotated_count == 0:
        errors.append(f"{split_name}: no annotated samples with non-empty query found.")

    if empty_ids:
        sample = ", ".join(empty_ids[:5])
        errors.append(f"{split_name}: {len(empty_ids)} samples have empty query (e.g. {sample}).")
    if bad_bbox_ids:
        sample = ", ".join(bad_bbox_ids[:5])
        errors.append(f"{split_name}: {len(bad_bbox_ids)} samples have invalid bbox (e.g. {sample}).")
    if missing_field_ids:
        sample = ", ".join(missing_field_ids[:5])
        errors.append(f"{split_name}: {len(missing_field_ids)} samples missing required fields (e.g. {sample}).")
    return errors


def flip_spatial_query(query: str) -> tuple[str, bool]:
    """Invert horizontal spatial terms in a query (left <-> right, leftmost <-> rightmost)."""
    def repl(m: re.Match) -> str:
        word = m.group(0)
        lower = word.lower()
        if lower == "left":
            replacement = "right"
        elif lower == "right":
            replacement = "left"
        elif lower == "leftmost":
            replacement = "rightmost"
        elif lower == "rightmost":
            replacement = "leftmost"
        elif lower == "left-most":
            replacement = "right-most"
        elif lower == "right-most":
            replacement = "left-most"
        else:
            return word
        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement.capitalize()
        return replacement

    pattern = re.compile(r"\b(left-most|right-most|leftmost|rightmost|left|right)\b", re.IGNORECASE)
    flipped, count = pattern.subn(repl, query)
    return flipped, count > 0


def count_effective_training_samples(data: dict, *, augment_flip: bool = True) -> int:
    """Count effective training samples produced by dataset after filtering and flip augmentation."""
    count = 0
    for item in data.values():
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        bbox = item.get("bbox")
        if not query or not bbox:
            continue
        count += 1
        if augment_flip:
            _, has_spatial = flip_spatial_query(query)
            if has_spatial:
                count += 1
    return count


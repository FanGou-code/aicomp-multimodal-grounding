"""Validation for synthetic English visual-grounding queries."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
_COORDINATES = re.compile(r"[\[(]\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?")
_CONTROL_TOKEN = re.compile(r"<\|[^<>\r\n]*\|>")
_FORBIDDEN = ("red rectangle", "red outline", "bounding box", "image 1", "image 2", "first image", "second image")


def clean_query_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().strip("`").strip().strip('"').strip("'").strip()


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


def preflight_check_dataset(data: dict, *, split_name: str = "dataset") -> list[str]:
    """Validate a training/val dataset dict before GPU training.

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

    required_fields = ("visible", "infrared", "depth", "query", "bbox")

    for key, item in data.items():
        for field in required_fields:
            if field not in item:
                missing_field_ids.append(key)
                break
        else:
            query = item.get("query", "")
            if not isinstance(query, str) or not query.strip():
                empty_ids.append(key)
            if validate_bbox(item.get("bbox")) is None:
                bad_bbox_ids.append(key)

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

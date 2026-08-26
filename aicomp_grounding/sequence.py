"""Sequence-level annotation planning and output validation."""

from __future__ import annotations

import json
import re

from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.query import clean_query_text, validate_generated_query, validate_query_style
from aicomp_grounding.sharding import group_keys_by_scene

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
_ANNOTATION_TERM = re.compile(
    r"\b(?:target|image|crop|annotation|annotated|bbox|coordinate|"
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


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Annotation response contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json_object(text: str, *, label: str) -> dict:
    if not isinstance(text, str):
        raise ValueError(f"{label} response must be text")
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
    try:
        payload = json.loads(candidate, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} response must be a JSON object")
    return payload


def parse_frame_query_candidates(text: str) -> dict[str, object]:
    payload = _parse_json_object(text, label="Frame query")
    expected = {"query", "alternate_query", "uncertain"}
    optional = {"style"}
    if set(payload) - expected - optional:
        raise ValueError(f"Frame query must contain exactly {sorted(expected)}")
    if "style" in payload and not isinstance(payload["style"], str):
        raise ValueError("Frame query style must be a string when present")
    query = clean_query_text(payload["query"])
    valid, reason = validate_annotation_query(query)
    if not valid:
        raise ValueError(f"Invalid primary query {query!r}: {reason}")
    valid, reason = validate_query_style(query)
    if not valid:
        raise ValueError(f"Primary query {query!r}: {reason}")
    alternate_value = payload["alternate_query"]
    alternate = None
    if alternate_value is not None:
        alternate = clean_query_text(alternate_value)
        valid, reason = validate_annotation_query(alternate)
        if not valid:
            # The alternate is a secondary candidate; a scaffolding violation
            # there should not sink the whole frame's valid primary query.
            print(
                f"Warning: dropping alternate query {alternate!r} "
                f"({reason}); keeping primary {query!r}",
                flush=True,
            )
            alternate = None
        elif alternate.casefold() == query.casefold():
            raise ValueError("Alternate query duplicates the primary query")
    if not isinstance(payload["uncertain"], bool):
        raise ValueError("Frame query uncertain must be a boolean")
    return {
        "query": query,
        "alternate_query": alternate,
        "uncertain": payload["uncertain"],
    }


def parse_verification_bbox(text: str) -> list[float] | None:
    """Parse a localization response into a normalized XYXY bbox or None.

    Accepts either a JSON object ``{"bbox": [x1, y1, x2, y2]}`` or a bare
    ``[x1, y1, x2, y2]`` array. Coordinates given in the 0-1000 integer
    convention (any value clearly above the unit range) are scaled to [0, 1].
    Returns None when no valid four-coordinate box can be extracted.
    """
    from aicomp_grounding.bbox import validate_bbox

    cleaned = clean_query_text(text)
    if not cleaned:
        return None

    candidates: list[list[float]] = []
    # Strip code fences if GLM wrapped the JSON in ```json ... ```.
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fenced:
        cleaned = fenced.group(1)

    # Try parsing as JSON first (object with bbox, or bare array).
    try:
        parsed = json.loads(cleaned, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        box = parsed.get("bbox") or parsed.get("box") or parsed.get("coordinates")
        if isinstance(box, list) and len(box) == 4:
            candidates.append([float(v) for v in box])
    elif isinstance(parsed, list) and len(parsed) == 4:
        candidates.append([float(v) for v in parsed])

    # Fall back to the first four-number run if JSON parsing failed.
    if not candidates:
        match = re.search(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
            r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            cleaned,
        )
        if match:
            candidates.append([float(g) for g in match.groups()])

    for box in candidates:
        if any(value > 1.5 for value in box):
            box = [value / 1000.0 for value in box]
        if validate_bbox(box) is not None:
            return [round(value, 6) for value in box]
    return None


def sequence_keys(dataset: dict, sequence_id: str) -> list[str]:
    return group_keys_by_scene(list(dataset), dataset).get(sequence_id, [])

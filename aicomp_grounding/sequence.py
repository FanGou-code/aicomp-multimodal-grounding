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
    raw_query = payload.get("final_query") or payload.get("query")
    if not isinstance(raw_query, str):
        raise ValueError("Frame query response must contain a 'final_query' or 'query' string")
    query = clean_query_text(raw_query)
    valid, reason = validate_annotation_query(query)
    if not valid:
        raise ValueError(f"Invalid primary query {query!r}: {reason}")
    valid, reason = validate_query_style(query)
    if not valid:
        raise ValueError(f"Primary query {query!r}: {reason}")
    uncertain = bool(payload.get("uncertain", False))
    return {
        "query": query,
        "alternate_query": None,
        "uncertain": uncertain,
        "target_category": payload.get("target_category"),
        "visible_attributes": payload.get("visible_attributes"),
        "action_or_state": payload.get("action_or_state"),
        "spatial_landmark": payload.get("spatial_landmark"),
        "disambiguation_cue": payload.get("disambiguation_cue"),
    }





def sequence_keys(dataset: dict, sequence_id: str) -> list[str]:
    return group_keys_by_scene(list(dataset), dataset).get(sequence_id, [])

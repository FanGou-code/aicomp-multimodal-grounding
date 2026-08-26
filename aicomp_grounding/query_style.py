"""Deterministic query-style planning and teacher prompt contracts.

The old annotation pipeline lets the teacher model choose the query wording
freely. The new strategy separates semantics from syntax: scene cards decide
which styles a sequence can support, a style plan assigns concrete slots, and
the generation prompt is restricted to one official template family.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics

from aicomp_grounding.sharding import extract_scene_id

QUERY_STYLE_GROUPS = (
    "ordinal",
    "spatial_landmark",
    "distance",
    "scene_location",
    "attribute_action",
)

STYLE_ALIASES = {
    "ordinal": "ordinal",
    "spatial": "spatial_landmark",
    "spatial_landmark": "spatial_landmark",
    "landmark": "spatial_landmark",
    "distance": "distance",
    "far": "distance",
    "location": "scene_location",
    "scene_location": "scene_location",
    "area": "scene_location",
    "attribute_action": "attribute_action",
    "descriptive": "attribute_action",
    "attribute": "attribute_action",
    "action": "attribute_action",
}

DEFAULT_QUERY_WEIGHTS = {
    "ordinal": 0.285,
    "spatial_landmark": 0.34,
    "distance": 0.105,
    "scene_location": 0.057,
    "attribute_action": 0.213,
}

TEMPLATE_FAMILIES = {
    "ordinal": (
        "FROM_LEFT_TO_RIGHT",
        "ORDINAL_POSITION",
        "EXTREME_POSITION",
    ),
    "spatial_landmark": (
        "RELATIVE_TO_LANDMARK",
        "BESIDE_ABOVE_BELOW",
        "IN_FRONT_BEHIND",
    ),
    "distance": (
        "DISTANCE_FROM_CAMERA",
        "FOREGROUND_BACKGROUND",
    ),
    "scene_location": (
        "SCENE_REGION",
        "REGION_CORNER",
    ),
    "attribute_action": (
        "ATTRIBUTE_ACTION",
        "ATTRIBUTE_CONTEXT",
    ),
}

FALLBACK_ORDER = (
    "ordinal",
    "spatial_landmark",
    "distance",
    "scene_location",
    "attribute_action",
)

ORDINAL_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|"
    r"last)\b",
    re.IGNORECASE,
)
SPATIAL_RE = re.compile(
    r"\b(?:left|right|beside|below|above|under|behind|front|"
    r"next to|in front of|near|opposite|between)\b",
    re.IGNORECASE,
)
DISTANCE_RE = re.compile(
    r"\b(?:distance|from the camera|foreground|background|"
    r"closest to the camera|farthest from)\b",
    re.IGNORECASE,
)
LOCATION_RE = re.compile(
    r"\b(?:middle|center|centre|corner|side|edge|area|region|"
    r"plaza|street|road|field|path|wall|ground|steps|door|window)\b",
    re.IGNORECASE,
)
ATTRIBUTE_ACTION_RE = re.compile(
    r"\b(?:red|blue|green|yellow|black|white|brown|gray|grey|pink|orange|"
    r"purple|dark|light|large|small|standing|sitting|walking|lying|running|"
    r"holding|carrying|wearing|parked|mounted|attached|hanging)\b",
    re.IGNORECASE,
)

SCENE_CARD_PROMPT = """You are analyzing one RGB scene for query-label planning.

The red rectangle marks the tracked object. Return only JSON:
{
  "scene_type": "plaza",
  "tracked_category": "person",
  "same_category_count": 2,
  "ordinal_position": "second from left to right",
  "stable_attributes": ["red jacket"],
  "action_or_state": "sitting",
  "landmark": "stone ledge",
  "scene_region": "right side of the plaza",
  "distance_hint": "foreground",
  "unique_in_scene": false
}

Rules:
- same_category_count counts clearly visible objects of the same category.
- ordinal_position is null unless same_category_count >= 3.
- landmark is null unless one object or area can be referenced reliably.
- distance_hint is one of null, foreground, middle distance, background, nearest, farthest.
- scene_region describes the target position inside the scene, not an image coordinate.
- Do not mention the red rectangle, target, image, frame, annotation, box, or coordinates.
Output JSON only."""

SCENE_CARD_PROMPT_HASH = hashlib.sha256(SCENE_CARD_PROMPT.encode("utf-8")).hexdigest()


def normalize_style(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    try:
        return STYLE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown query style {value!r}") from exc


def classify_query(query: str) -> str:
    if DISTANCE_RE.search(query):
        return "distance"
    if ORDINAL_RE.search(query):
        return "ordinal"
    if SPATIAL_RE.search(query):
        return "spatial_landmark"
    if LOCATION_RE.search(query):
        return "scene_location"
    return "attribute_action"


def analyze_queries(queries: list[str]) -> dict[str, object]:
    if not queries:
        raise ValueError("Query list is empty")
    word_counts = [len(query.split()) for query in queries]
    group_counts = {style: 0 for style in QUERY_STYLE_GROUPS}
    for query in queries:
        group_counts[classify_query(query)] += 1
    total = len(queries)
    return {
        "count": total,
        "mean_words": statistics.mean(word_counts),
        "median_words": statistics.median(word_counts),
        "group_counts": group_counts,
        "group_ratios": {
            style: group_counts[style] / total for style in QUERY_STYLE_GROUPS
        },
    }


def parse_scene_card(text: str) -> dict:
    cleaned = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene card is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Scene card must be a JSON object")
    expected = {
        "scene_type",
        "tracked_category",
        "same_category_count",
        "ordinal_position",
        "stable_attributes",
        "action_or_state",
        "landmark",
        "scene_region",
        "distance_hint",
        "unique_in_scene",
    }
    if set(payload) != expected:
        raise ValueError(
            f"Scene card must contain exactly {sorted(expected)}, got {sorted(payload)}"
        )
    if not isinstance(payload["scene_type"], str) or not payload["scene_type"]:
        raise ValueError("Scene card scene_type must be non-empty")
    if not isinstance(payload["tracked_category"], str) or not payload["tracked_category"]:
        raise ValueError("Scene card tracked_category must be non-empty")
    if isinstance(payload["same_category_count"], bool) or not isinstance(
        payload["same_category_count"], int
    ):
        raise ValueError("Scene card same_category_count must be an integer")
    if isinstance(payload["unique_in_scene"], bool) is False:
        raise ValueError("Scene card unique_in_scene must be a boolean")
    for field in ("stable_attributes",):
        value = payload[field]
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"Scene card {field} must be a list of strings")
    for field in (
        "ordinal_position",
        "action_or_state",
        "landmark",
        "scene_region",
        "distance_hint",
    ):
        if payload[field] is not None and not isinstance(payload[field], str):
            raise ValueError(f"Scene card {field} must be a string or null")
    return payload


def extract_supported_styles(card: dict) -> list[str]:
    supported: list[str] = []
    if card.get("same_category_count", 0) >= 2 and card.get("ordinal_position"):
        supported.append("ordinal")
    if card.get("landmark"):
        supported.append("spatial_landmark")
    if card.get("distance_hint"):
        supported.append("distance")
    if card.get("scene_region"):
        supported.append("scene_location")
    supported.append("attribute_action")
    return list(dict.fromkeys(supported))


def build_style_plan(
    dataset: dict,
    scene_cards: dict,
    *,
    seed: int = 42,
    queries_per_frame: int = 3,
    target_weights: dict[str, float] | None = None,
) -> dict:
    if not 1 <= queries_per_frame <= 6:
        raise ValueError("queries_per_frame must be between 1 and 6")
    if target_weights is None:
        target_weights = dict(DEFAULT_QUERY_WEIGHTS)
    if set(target_weights) != set(QUERY_STYLE_GROUPS) or abs(sum(target_weights.values()) - 1.0) > 1e-6:
        raise ValueError("target_weights must cover every query style group and sum to 1")

    slots: list[tuple[str, dict, str]] = []
    for sample_id, item in dataset.items():
        sequence_id = extract_scene_id(sample_id, item)
        card = scene_cards.get(sequence_id, {})
        supported = extract_supported_styles(card)
        for rank in range(1, queries_per_frame + 1):
            slots.append((sample_id, {"rank": rank, "supported": supported}, sequence_id))

    rng = random.Random(seed)
    shuffled = list(slots)
    rng.shuffle(shuffled)

    remaining = dict(target_weights)
    items: dict[str, dict] = {}
    for sample_id, slot, sequence_id in shuffled:
        supported = slot["supported"]
        normalized = {
            style: weight
            for style, weight in remaining.items()
            if style in supported and weight > 0
        }
        if not normalized:
            normalized = {
                style: target_weights[style]
                for style in supported
            }
        if not normalized or sum(normalized.values()) <= 0:
            normalized = {"attribute_action": 1.0}
        style = rng.choices(list(normalized), weights=list(normalized.values()), k=1)[0]
        remaining[style] = max(0.0, remaining[style] - 1.0 / max(len(shuffled), 1))

        template_family = TEMPLATE_FAMILIES[style][slot["rank"] % len(TEMPLATE_FAMILIES[style])]
        fallback = next(
            (candidate for candidate in FALLBACK_ORDER if candidate in supported and candidate != style),
            "attribute_action",
        )
        key = f"{sample_id}_q{slot['rank']}" if queries_per_frame > 1 else sample_id
        items[key] = {
            "source_sample_id": sample_id,
            "sequence_id": sequence_id,
            "rank": slot["rank"],
            "style": style,
            "template_family": template_family,
            "fallback_style": fallback,
            "min_words": 7,
            "max_words": 13,
        }
    return {
        "protocol_version": 1,
        "seed": seed,
        "queries_per_frame": queries_per_frame,
        "target_weights": dict(target_weights),
        "items": items,
    }


def style_plan_fingerprint(plan: dict) -> str:
    return stable_json_hash(plan["items"])


def expand_annotation_source(dataset: dict, plan: dict) -> dict:
    expanded: dict[str, dict] = {}
    for sample_id, spec in plan["items"].items():
        source = dataset[spec["source_sample_id"]]
        if sample_id in expanded:
            raise ValueError(f"Duplicate expanded sample ID {sample_id!r}")
        item = dict(source)
        item["annotation_style"] = spec["style"]
        item["annotation_style_family"] = spec["template_family"]
        item["annotation_min_words"] = spec["min_words"]
        item["annotation_max_words"] = spec["max_words"]
        item["annotation_fallback_style"] = spec["fallback_style"]
        expanded[sample_id] = item
    return expanded


def build_style_prompt(
    style: str,
    *,
    min_words: int,
    max_words: int,
    template_family: str,
    fallback_style: str,
) -> str:
    style = normalize_style(style)
    restrictions = {
        "ordinal": (
            "- Use counting only when the scene contains a clearly visible group.\n"
            '- Use "From left to right, the {ordinal} {object}" or an extreme-position phrase.\n'
            "- Do not invent an ordinal when counting is uncertain."
        ),
        "spatial_landmark": (
            "- Reference one clearly visible landmark or scene object.\n"
            '- Use "to the right of", "beside", "behind", "in front of", or similar.\n'
            "- Do not count objects or mention the camera."
        ),
        "distance": (
            '- Use "nearest", "farthest", "closest", "foreground", or "background".\n'
            "- Avoid landmark relations and ordinal counting."
        ),
        "scene_location": (
            '- Use a scene region such as "middle of the plaza" or "top-right corner of the wall".\n'
            "- Do not mention image/frame coordinates or annotation terms."
        ),
        "attribute_action": (
            "- Combine visible attributes with action or location when available.\n"
            "- A short descriptive clause is acceptable only when the object is unique."
        ),
    }
    parent_rule = (
        "Requested style is ordinal; if the scene does not support counting, "
        f"fall back to {fallback_style} and return that style with the query."
        if style == "ordinal"
        else f"Requested style is {style}; if it is impossible in this scene, fall back to {fallback_style}."
    )
    return (
        "Write one official English visual-grounding query for the object in the red rectangle.\n"
        "The rectangle is an internal pointer; never mention the rectangle, target, image, frame, annotation, box, or coordinates.\n"
        f"Template family: {template_family}\n"
        f"Word count: {min_words}-{max_words} words.\n"
        f"{restrictions[style]}\n"
        f"{parent_rule}\n"
        "Return exactly:\n"
        '{"query":"...","alternate_query":null,"uncertain":false,"style":"..."}\n'
        "Output JSON only."
    )


def stable_json_hash(value: object, length: int = 64) -> str:
    """Small local helper to avoid importing artifact hashing in prompts."""
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return digest[:length]


STYLE_PROMPT_HASH = hashlib.sha256(
    "\n".join(
        build_style_prompt(
            style,
            min_words=7,
            max_words=13,
            template_family=TEMPLATE_FAMILIES[style][0],
            fallback_style="attribute_action",
        )
        for style in QUERY_STYLE_GROUPS
    ).encode("utf-8")
).hexdigest()

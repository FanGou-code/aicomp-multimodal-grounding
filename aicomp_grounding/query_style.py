"""Query style classification, audit metrics, and prompt contracts."""

from __future__ import annotations

import hashlib
import re
import statistics

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

DISAMBIGUATION_QUERY_PROMPT = """Write a natural English visual-grounding query for the physical object enclosed by the red rectangle.

The red rectangle is an external indicator only. Never mention the rectangle, marking, image, frame, annotation, coordinates, or target, and never mistake the red boundary color for the target's actual color.

Analyze the scene and return JSON:
{
  "target_category": "specific basic-level category of the tracked object",
  "visible_attributes": "salient color, clothing, material, or visual patterns",
  "action_or_state": "participle action, state, or posture",
  "spatial_landmark": "adjacent physical landmark, mount, or background structure; or null if open space",
  "disambiguation_cue": "spatial disambiguation index ONLY when multiple same-category objects exist (specify its sequence order, extreme boundary, depth row, or relative direction); or null if unique",
  "final_query": "natural compact English noun phrase of 6-20 words combining the non-null elements"
}

Query rules:
- Form: Write one compact English noun phrase. Use participle or prepositional phrases (e.g., 'standing on the grass', 'mounted on the wall', 'in a blue shirt') instead of relative clauses (avoid 'who is...', 'which was...').
- Article & capitalization: Begin with an article ('The', 'A', or 'An'); use 'The' for a specific instance.
- Punctuation: Do NOT end with a period or trailing punctuation.
- Categories: Use specific basic-level categories; only fallback to broad terms ('animal', 'vehicle', 'object') if the target is genuinely too distant or blurry to identify.
- Dual-track disambiguation:
  * Single target in scene: set disambiguation_cue to null; describe target with its intrinsic attributes, action, and physical location.
  * Multiple same-category objects in scene: you MUST provide a natural spatial disambiguation cue (such as its sequence order along a line/group, extreme boundary on the left/right/top/bottom, relative depth in the front/back row, or relative direction to neighboring objects) to uniquely distinguish it.
- Perspective: Express all directions strictly from the viewer's 2D perspective of the image.
Output JSON only."""

DISAMBIGUATION_PROMPT_HASH = hashlib.sha256(
    DISAMBIGUATION_QUERY_PROMPT.encode("utf-8")
).hexdigest()

# Retain backward-compatible alias for prompt hash referencing
STYLE_PROMPT_HASH = DISAMBIGUATION_PROMPT_HASH


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

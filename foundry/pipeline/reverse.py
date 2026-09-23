"""Reverse pass: query text -> box, through the same kernel the forward side uses.

Code parses the query into the shared intermediate form, enumerates the
category, sorts it in code and takes the k-th. The schema is
``{category, k, axis, direction, count}``.

Two stages, each with its own decoding:

``parse``       text only, greedy (``do_sample=false``), JSON mode.
``enumerate``   the image, thinking on, provider-default temperature.

A query whose axis needs the depth or infrared image (this pass sends neither)
or whose enumeration fails its own gates falls back to ``direct``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

AXES = ("x", "y", "area")
DIRECTIONS = ("asc", "desc")


def _load_prompt(name: str, default: str) -> str:
    try:
        path = Path(__file__).resolve().parents[2] / "configs" / "default" / "prompts" / f"{name}.md"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return default


_PARSE_DEFAULT = """Parse this English query for a visual grounding task and output its intent as JSON.

Query: {query}

Rules:
- "category": the shortest common noun naming the object to locate. Lower case, singular. Do not include color, size, position, material, or any attribute. If the query locates a part of an object, name the whole object.
- "selection": how the target is picked out among several instances of that category.
  - {"mode": "unique"} when the query names one specific object, with no ordering and no comparison to another instance.
  - {"mode": "rank", "k": <int>, "axis": <axis>, "direction": <"asc" | "desc">} when the query picks one instance by an ordered position.

Axis must be exactly one of:
  "x"     left/right position in the image   (leftmost = asc, rightmost = desc)
  "y"     top/bottom position in the image   (topmost  = asc, bottommost = desc)
  "area"  size in the image                  (smallest = asc, largest = desc)

k counts instances of "category" along that axis; k = 1 is the first in "direction".
When no ordering is expressed, use "unique"; never invent a rank.

Output JSON only:
{"category": "<noun>", "selection": {"mode": "unique"}}
{"category": "<noun>", "selection": {"mode": "rank", "k": 3, "axis": "x", "direction": "asc"}}"""

_ENUMERATE_DEFAULT = """One RGB image.

Task: list EVERY {category} visible in this image. Include small, distant, blurry, partially occluded, and instances cut off by the image edge. Do not stop at any number. Count first, then list.

Order does not matter; give each a tight bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.

Output JSON only:
{"count": <int>, "objects": [{"bbox": [x1, y1, x2, y2]}, ...]}"""

_DIRECT_DEFAULT = """Locate the object this query refers to in the image.

Query: {query}

Give one tight bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.

Output JSON only:
{"bbox": [x1, y1, x2, y2]}"""

PARSE_PROMPT = _load_prompt("parse", _PARSE_DEFAULT)
ENUMERATE_PROMPT = _load_prompt("enumerate", _ENUMERATE_DEFAULT)
DIRECT_PROMPT = _load_prompt("direct", _DIRECT_DEFAULT)

PROMPT_HASHES = {
    name: hashlib.sha256(text.encode("utf-8")).hexdigest()
    for name, text in (("parse", PARSE_PROMPT), ("enumerate", ENUMERATE_PROMPT), ("direct", DIRECT_PROMPT))
}


def _json_object(text: object) -> dict | None:
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_intent_response(text: object) -> dict | None:
    """``{"category", "mode", ...}`` or None when the reply is not that."""
    payload = _json_object(text)
    if payload is None or set(payload) != {"category", "selection"}:
        return None
    category = payload["category"]
    if not isinstance(category, str) or not category.strip():
        return None
    selection = payload["selection"]
    if not isinstance(selection, dict):
        return None
    mode = selection.get("mode")
    if mode == "unique":
        return {"category": category.strip().lower(), "mode": "unique"}
    if mode != "rank":
        return None
    k = selection.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        return None
    axis = selection.get("axis")
    if axis not in AXES:
        return None
    direction = selection.get("direction")
    if direction not in DIRECTIONS:
        return None
    return {
        "category": category.strip().lower(),
        "mode": "rank",
        "k": k,
        "axis": axis,
        "direction": direction,
    }


def _bbox(raw: object) -> list[float] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    if not all(0.0 <= value <= 1.0 for value in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def parse_enumeration_response(text: object) -> dict | None:
    """``{"count", "objects"}`` with the count gate applied, else None."""
    payload = _json_object(text)
    if payload is None or set(payload) != {"count", "objects"}:
        return None
    count = payload["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    objects = payload["objects"]
    if not isinstance(objects, list):
        return None
    boxes: list[list[float]] = []
    for item in objects:
        if not isinstance(item, dict) or set(item) != {"bbox"}:
            return None
        box = _bbox(item["bbox"])
        if box is None:
            return None
        boxes.append(box)
    if count != len(boxes):
        return None
    return {"count": count, "boxes": boxes}


def parse_direct_response(text: object) -> list[float] | None:
    payload = _json_object(text)
    if payload is None or set(payload) != {"bbox"}:
        return None
    return _bbox(payload["bbox"])


def axis_value(axis: str, bbox: list[float]) -> float:
    """Sort value of one box on one axis; all three come from the box alone."""
    if axis == "x":
        return bbox[0]
    if axis == "y":
        return bbox[1]
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def pick_kth(boxes: list[list[float]], *, k: int, axis: str, direction: str) -> list[float] | None:
    """The k-th box along ``axis``; None when there are fewer than k of them."""
    if k > len(boxes):
        return None
    ordered = sorted(boxes, key=lambda box: (axis_value(axis, box), box[0], box[1]))
    if direction == "desc":
        ordered.reverse()
    return list(ordered[k - 1])


def parse_messages(query: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise query parser. Return only valid JSON."},
        {"role": "user", "content": [{"type": "text", "text": PARSE_PROMPT.replace("{query}", query)}]},
    ]


def enumerate_messages(image_url: str, category: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise visual enumerator. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ENUMERATE_PROMPT.replace("{category}", category)},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        },
    ]


def direct_messages(image_url: str, query: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise visual locator. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DIRECT_PROMPT.replace("{query}", query)},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        },
    ]

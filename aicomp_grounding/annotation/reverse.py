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

from aicomp_grounding.ordinal_kernel import axis_value as kernel_axis_value
from aicomp_grounding.ordinal_kernel import order_indices

AXES = ("x", "y", "area")
DIRECTIONS = ("asc", "desc")


def read_prompt(name: str) -> str:
    """Load ``prompts/<name>.md``; a missing file is an error, not a fallback."""
    path = Path(__file__).resolve().parent / "prompts" / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


PARSE_PROMPT = read_prompt("parse")
ENUMERATE_PROMPT = read_prompt("enumerate")
DIRECT_PROMPT = read_prompt("direct")

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
    """Sort value of one box on one axis; all three come from the box alone.

    The parse stage only emits box-owned axes, so anything else is an error
    rather than a missing-data ``None`` from the shared kernel.
    """
    value = kernel_axis_value(axis, bbox)
    if value is None:
        raise ValueError(f"Unsupported axis: {axis!r}")
    return value


def pick_kth(boxes: list[list[float]], *, k: int, axis: str, direction: str) -> list[float] | None:
    """The k-th box along ``axis``; None when k is out of range.

    Ties keep the box tuple ascending whatever the direction, matching
    ``aicomp_grounding.serving.ordinal.resolve.rank_instances``.
    """
    if k < 1 or k > len(boxes):
        return None
    values = [axis_value(axis, box) for box in boxes]
    order = order_indices(values, boxes, direction=direction)
    return list(boxes[order[k - 1]])


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

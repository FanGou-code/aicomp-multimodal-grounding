"""Turn one target's facts into one English sentence.

Everything that is true of the target inside its frame is written out and handed
over; nothing selects or prioritises among those facts. The teacher reads the
image and the list, and decides which of them to voice and how. Wording is not
checked: whatever comes back goes to human review, which is the only review
pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from foundry.pipeline.facts import ObjectFacts

#: Which way a direction reads.
DIRECTION_SIDE = {"asc": "left", "desc": "right"}

#: Heads that are not described by a bare color: "the white person" reads as a
#: race descriptor. These are described by a feature or a position instead.
COLOR_SUPPRESSED_HEADS = frozenset(
    {"person", "child", "man", "woman", "men", "women", "people", "couple"}
)


def _load_prompt(name: str, default: str) -> str:
    try:
        path = Path(__file__).resolve().parents[2] / "configs" / "default" / "prompts" / f"{name}.md"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return default


_REALIZE_DEFAULT = """The image has a box drawn around one object.

Write ONE short English noun phrase that identifies exactly that object and nothing else in the image. Everything listed below is true of it; use whichever of those facts you need, and as many as you need, to make the phrase unambiguous.

Use only the facts listed below. Do not add any property that is not listed. Do not refer to the image, the frame, the box, or the rectangle. Do not mention that you were given facts.

Output JSON only:
{"query": "<the noun phrase>"}"""

REALIZE_PROMPT = _load_prompt("realize", _REALIZE_DEFAULT)
REALIZE_PROMPT_HASH = hashlib.sha256(REALIZE_PROMPT.encode("utf-8")).hexdigest()

#: Decoding for the wording call: thinking off, provider default temperature.
REALIZE_STAGE = {
    "max_tokens": 256,
    "temperature": None,
    "do_sample": None,
    "thinking_mode": "disabled",
    "response_format": None,
}


def color_field(target: ObjectFacts) -> str | None:
    """The color to hand over; None for suppressed heads."""
    if target.head in COLOR_SUPPRESSED_HEADS:
        return None
    return target.color


def fact_lines(target: ObjectFacts) -> list[str]:
    """Every fact that is true of ``target`` inside its frame, one per line.

    No selection and no priority: what gets said is the teacher's call.
    """
    lines = [f"- the object is a: {target.head}"]

    color = color_field(target)
    if color:
        lines.append(f"- its color is: {color}")
    if target.features:
        lines.append(f"- notable features: {target.features}")

    if target.count_in_head >= 2:
        lines.append(f"- the frame contains {target.count_in_head} of that category")
    if target.rank_left is not None:
        lines.append(f"- counting from the left, it is number {target.rank_left}")
    if target.rank_right is not None:
        lines.append(f"- counting from the right, it is number {target.rank_right}")

    if target.side_of_image:
        lines.append(f"- it is on the {target.side_of_image} side of the image")

    for side, attr in (("left", "anchors_left"), ("right", "anchors_right")):
        for _index, category in getattr(target, attr):
            lines.append(f"- it is on the {side} side of the {category}")

    for flag, text in (
        ("is_leftmost", "no other object in the frame is further left"),
        ("is_rightmost", "no other object in the frame is further right"),
        ("is_topmost", "no other object in the frame is higher up"),
        ("is_bottommost", "no other object in the frame is lower down"),
        ("is_closest", "no other object in the frame is closer to the camera"),
        ("is_farthest", "no other object in the frame is farther from the camera"),
    ):
        if getattr(target, flag):
            lines.append(f"- {text}")

    if target.is_in_foreground:
        lines.append("- it is in the foreground")
    if target.is_in_background:
        lines.append("- it is in the background")
    if target.median_mm is not None:
        lines.append(f"- it is about {target.median_mm} mm from the camera")
    return lines


def facts_prompt_text(target: ObjectFacts) -> str:
    """The fact list as it is handed to the teacher."""
    return "\n".join(fact_lines(target))


def realize_messages(image_url: str, target: ObjectFacts) -> list[dict]:
    body = REALIZE_PROMPT + "\n\nFacts:\n" + facts_prompt_text(target)
    return [
        {"role": "system", "content": "You write one short, natural English noun phrase. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "One RGB scene with a box drawn around the target object."},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                {"type": "text", "text": body},
            ],
        },
    ]


def parse_realize_response(text: object) -> str | None:
    """Return the one query string, or None when the reply is not that."""
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
    if not isinstance(payload, dict) or set(payload) != {"query"}:
        return None
    query = payload["query"]
    if not isinstance(query, str) or not query.strip():
        return None
    return " ".join(query.split())

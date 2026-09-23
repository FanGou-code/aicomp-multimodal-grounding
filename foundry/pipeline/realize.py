"""Turn one target's chosen facts into one English sentence.

Code decides which dimension disambiguates the target inside its own frame; the
teacher words it. The prompt carries facts only -- no sentence template and no
worked example. How the sentence is phrased is not checked: whatever comes back
goes to human review, which is the only review pass.
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

Write ONE short English noun phrase that identifies exactly that object and nothing else in the image.

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


def rank_and_direction(target: ObjectFacts) -> tuple[int, str] | None:
    """The rank to name: the smaller of the two, with its direction.

    Ties go to "from the left".
    """
    if target.rank_left is None or target.rank_right is None:
        return None
    if target.rank_left <= target.rank_right:
        return target.rank_left, "asc"
    return target.rank_right, "desc"


def choose_dimension(target: ObjectFacts, frame: list[ObjectFacts]) -> dict | None:
    """Pick the one dimension that names ``target`` and nothing else in ``frame``.

    Dimensions are tried in a fixed order; the first that exists and is unique
    wins. Uniqueness is counted over same-head objects.
    """
    peers = [o for o in frame if o.head == target.head]
    rank_direction = rank_and_direction(target)

    if rank_direction is not None:
        rank, direction = rank_direction
        return {
            "family": "ordinal_direction",
            "facts": (f"rank:{rank}", direction),
            "fields": {
                "category": target.head,
                "k": rank,
                "axis": "x",
                "direction": direction,
                "count": target.count_in_head,
                "color": color_field(target),
                "feature": target.features,
            },
        }

    for side, attr in (("left", "anchors_left"), ("right", "anchors_right")):
        for anchor in getattr(target, attr):
            # The anchor head is unique in the frame (facts.py guarantees it);
            # the phrase is unambiguous only if this target is the sole
            # same-head object on that side of that anchor.
            if sum(1 for o in peers if anchor in getattr(o, attr)) == 1:
                return {
                    "family": "side_of_anchor",
                    "facts": (f"anchor:{side}:{anchor[0]}",),
                    "fields": {
                        "category": target.head,
                        "anchor": anchor[1],
                        "anchor_side": side,
                        "color": color_field(target),
                        "feature": target.features,
                    },
                }

    for flag, family, fact in (
        ("is_closest", "superlative_camera", "y2-max"),
        ("is_farthest", "superlative_camera", "y2-min"),
        ("is_leftmost", "superlative_camera", "x-min"),
        ("is_rightmost", "superlative_camera", "x-max"),
        ("is_topmost", "superlative_camera", "y-min"),
        ("is_bottommost", "superlative_camera", "y-max"),
    ):
        if getattr(target, flag):
            return {
                "family": family,
                "facts": (fact,),
                "fields": {
                    "category": target.head,
                    "position": fact,
                    "color": color_field(target),
                    "feature": target.features,
                },
            }

    for flag, band in (("is_in_foreground", "foreground"), ("is_in_background", "background")):
        if getattr(target, flag) and sum(1 for o in peers if getattr(o, flag)) == 1:
            return {
                "family": "superlative_camera",
                "facts": (band,),
                "fields": {
                    "category": target.head,
                    "depth_band": band,
                    "color": color_field(target),
                    "feature": target.features,
                },
            }

    if target.side_of_image and sum(1 for o in peers if o.side_of_image == target.side_of_image) == 1:
        return {
            "family": "side_of_anchor",
            "facts": (f"image:{target.side_of_image}",),
            "fields": {
                    "category": target.head,
                    "side_of_image": target.side_of_image,
                    "color": color_field(target),
                    "feature": target.features,
                },
        }

    if target.color and color_field(target) is not None and sum(
        1 for o in peers if o.color is not None and o.color == target.color
    ) == 1:
        return {
            "family": "plain_attribute",
            "facts": ("color",),
            "fields": {
                "category": target.head,
                "color": color_field(target),
                "feature": target.features,
            },
        }

    if target.features and sum(
        1 for o in peers if o.features is not None and o.features == target.features
    ) == 1:
        return {
            "family": "plain_attribute",
            "facts": ("feature",),
            "fields": {"category": target.head, "feature": target.features},
        }

    if len(peers) == 1:
        return {
            "family": "plain_attribute",
            "facts": ("unique-category",),
            "fields": {"category": target.head},
        }
    return None


def dimension_prompt_text(dimension: dict) -> str:
    """Render a dimension as the fact list the teacher is allowed to use.

    Numbers stay numbers: the teacher picks the wording (``third``, ``3rd``,
    ``number 3``) and nothing here constrains it.
    """
    fields = dimension["fields"]
    labels = {
        "category": "the object is a",
        "k": "its position",
        "direction": "counting",
        "position": "it is the",
        "depth_band": "it is in the",
        "side_of_image": "it is on the",
        "anchor": "it is on the",
        "color": "its color is",
        "feature": "one notable feature:",
    }
    lines = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        if key in ("axis", "count", "anchor_side"):
            continue
        if key == "direction":
            value = "from the " + DIRECTION_SIDE[value]
        if key == "anchor":
            value = f"{fields['anchor_side']} side of the {value}"
        if key == "position":
            value = {
                "y2-max": "closest to the camera",
                "y2-min": "farthest from the camera",
                "x-min": "leftmost one",
                "x-max": "rightmost one",
                "y-min": "topmost one",
                "y-max": "bottommost one",
            }[value]
        lines.append(f"- {labels.get(key, key)}: {value}")
    return "\n".join(lines)


def realize_messages(image_url: str, dimension: dict) -> list[dict]:
    body = REALIZE_PROMPT + "\n\nFacts:\n" + dimension_prompt_text(dimension)
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

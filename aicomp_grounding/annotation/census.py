"""Census protocol: one enumeration call per frame, code validates the envelope.

Pass (prompt in ``configs/default/prompts/``):
- ``findall`` once per frame: recognise the red-boxed category, count how many
  instances of it the frame holds, then branch --
  * two or more: list every instance of that category, no cap
  * exactly one (the red-boxed object itself): skip that category and list
    other objects in the frame the teacher is confident about instead
  Either branch reports the count it listed.

Deterministic gates in code (the teacher never self-certifies):
- schema: ``{mode, category, count, objects}``, indices sequential 1..N
- self-reported ``count`` must equal the number of listed objects
- everything else is normalized in code, never fatal: ordering -> sorted by x1
  and renumbered; near-identical boxes (IoU >= 0.95) -> dedup; zero-area boxes
  -> dropped

The red box names the category to start from; it is not re-found and not
verified against.

Prompts contain no example queries and no style options.
"""

from __future__ import annotations

import hashlib
import json
import re

from aicomp_grounding.bbox import compute_iou

def _load_prompt(name: str, default: str) -> str:
    """Load a prompt from configs/default/prompts/<name>.md, falling back to default."""
    try:
        from pathlib import Path
        config_path = Path(__file__).resolve().parents[2] / "configs" / "default" / "prompts" / f"{name}.md"
        if config_path.is_file():
            return config_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return default

_FINDALL_DEFAULT = """The red rectangle marks one reference object in the scene.

Step 1. Name the category of the object inside the red rectangle, then count how many instances of that category the whole frame contains.

Step 2. Branch on that count:
- If the frame contains TWO OR MORE instances of that category: list EVERY instance of it — small, distant, blurry, partially occluded, cut off by the image edge. Do not stop at any number. Do not cap the list.
- If the frame contains EXACTLY ONE instance (the reference object itself): do not list that category. Instead list the other objects in the frame you are most confident about — clear outline, nameable at a glance, any category. Skip tiny clutter and anything you cannot identify precisely.

Order the list from left to right. Number them 1..N. For each object give a short common category name and its bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.

Output JSON only:
{"mode": "instances" or "other", "category": "<the red-boxed category>", "count": <int>, "objects": [{"i": 1, "category": "<category name>", "bbox": [x1, y1, x2, y2]}, ...]}

"mode" is "instances" when you listed every instance of the red-boxed category, "other" when you listed other confident objects instead. "count" must equal the number of objects you listed."""

_ATTR_DEFAULT = """The image shows numbered boxes around objects in the scene.
For each numbered object report only what is directly visible: its color and one notable visible feature.
Do not guess occluded or unclear properties.
Output JSON only:
{"1": {"color": "...", "features": "..."}, ...}"""

FINDALL_PROMPT = _load_prompt("findall", _FINDALL_DEFAULT)
ATTR_PROMPT = _load_prompt("attr", _ATTR_DEFAULT)

FINDALL_PROMPT_HASH = hashlib.sha256(FINDALL_PROMPT.encode("utf-8")).hexdigest()
ATTR_PROMPT_HASH = hashlib.sha256(ATTR_PROMPT.encode("utf-8")).hexdigest()

SELF_DUP_IOU = 0.95
BBOX_SLACK = 0.02

#: Which branch the enumeration took; ``instances`` means "I listed every
#: instance of the reference category", ``other`` means "that category was a
#: single object, so I listed other confident objects instead".
ENUM_MODES = ("instances", "other")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Census response contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json_object(text: object, *, label: str) -> dict:
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


def _scale_bbox(raw: object, scale: tuple[float, float]) -> list[float] | None:
    """Validate one raw bbox and convert it to normalized [0, 1] via (sx, sy).

    Returns ``None`` for zero-area boxes: degeneracy (x1 == x2 or y1 == y2)
    is preserved by uniform scaling, so such an entry is unusable under every
    convention and is dropped instead of killing the response.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("census bbox must be [x1, y1, x2, y2]")
    values = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("census bbox coordinates must be numbers")
        values.append(float(value))
    sx, sy = scale
    values = [values[0] * sx, values[1] * sy, values[2] * sx, values[3] * sy]
    if any(not -BBOX_SLACK <= value <= 1.0 + BBOX_SLACK for value in values):
        raise ValueError(f"census bbox normalizes outside [0, 1]: {values!r}")
    values = [min(1.0, max(0.0, value)) for value in values]
    if values[2] - values[0] < 0.001 or values[3] - values[1] < 0.001:
        return None
    return values


def _normalize_objects(cleaned: list[dict]) -> list[dict]:
    """Sort by x1, drop near-identical duplicates, renumber 1..N.

    All steps are invariant under the uniform scaling that separates the
    coordinate conventions, so ordering and dedup stay convention-agnostic.
    """
    cleaned.sort(key=lambda item: item["bbox"][0])
    kept: list[dict] = []
    for item in cleaned:
        if any(compute_iou(prev["bbox"], item["bbox"]) >= SELF_DUP_IOU for prev in kept):
            continue
        kept.append(item)
    for index, item in enumerate(kept, start=1):
        item["i"] = index
    return kept


def parse_findall_response(
    text: object,
) -> dict:
    """Validate one enumeration response through every deterministic gate.

    The response carries the branch it took, the reference category, the count
    it claims, and the objects.  The teacher's coordinate convention is
    auto-detected: values within [0, 1] are used directly, otherwise the
    per-mille (0-1000, GLM/Qwen family convention) is applied. The prompt asks
    for normalized fractions and forbids pixels, so there is no pixel
    convention to fall back on.
    """
    payload = _parse_json_object(text, label="Census findall")
    if set(payload) != {"mode", "category", "count", "objects"}:
        raise ValueError("Census findall schema must be exactly {mode, category, count, objects}")
    mode = payload["mode"]
    if mode not in ENUM_MODES:
        raise ValueError(f"Census findall mode must be one of {ENUM_MODES}")
    category = payload["category"]
    if not isinstance(category, str) or not category.strip() or len(category) > 64:
        raise ValueError("Census findall category is missing or oversized")
    count = payload["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Census findall count must be a non-negative integer")
    objects = payload["objects"]
    if not isinstance(objects, list) or not objects:
        raise ValueError("Census findall must list at least one object")
    if count != len(objects):
        raise ValueError(
            f"Census findall count {count} does not match the {len(objects)} objects listed"
        )
    raw_boxes: list[dict] = []
    for expected_i, item in enumerate(objects, start=1):
        if not isinstance(item, dict) or set(item) != {"i", "category", "bbox"}:
            raise ValueError("Census object entries must be exactly {i, category, bbox}")
        index = item["i"]
        if isinstance(index, bool) or not isinstance(index, int) or index != expected_i:
            raise ValueError("Census object indices must be sequential 1..N")
        name = item["category"]
        if not isinstance(name, str) or not name.strip() or len(name) > 64:
            raise ValueError(f"Census object {index} category is missing or oversized")
        raw_boxes.append({"i": index, "category": name.strip(), "bbox": item["bbox"]})

    flat = [value for item in raw_boxes for value in item["bbox"]]
    numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in flat
    )
    if numeric and all(-BBOX_SLACK <= value <= 1.0 + BBOX_SLACK for value in flat):
        candidates = [("normalized-0-1", (1.0, 1.0))]
    else:
        candidates = [("per-mille-0-1000", (1.0 / 1000.0, 1.0 / 1000.0))]

    errors: list[str] = []
    for name, scale in candidates:
        try:
            cleaned: list[dict] = []
            for item in raw_boxes:
                bbox = _scale_bbox(item["bbox"], scale)
                if bbox is None:
                    continue
                cleaned.append({"i": item["i"], "category": item["category"], "bbox": bbox})
        except ValueError as exc:
            errors.append(f"[{name}] {exc}")
            continue
        if not cleaned:
            errors.append(f"[{name}] census response has no non-degenerate boxes")
            continue
        cleaned = _normalize_objects(cleaned)
        return {
            "mode": mode,
            "category": category.strip(),
            "count": count,
            "objects": cleaned,
            "bbox_convention": name,
        }
    raise ValueError(
        "Census response failed under every coordinate convention: " + " | ".join(errors)
    )


def parse_attr_response(text: object, *, indices: list[int]) -> dict:
    """Validate one attr response against the requested numbered objects."""
    payload = _parse_json_object(text, label="Census attr")
    expected = {str(i) for i in indices}
    if set(payload) != expected:
        raise ValueError(
            f"Census attr must cover exactly the numbered objects {sorted(expected)}"
        )
    result: dict[str, dict] = {}
    for key in sorted(expected, key=int):
        value = payload[key]
        if not isinstance(value, dict) or set(value) != {"color", "features"}:
            raise ValueError(f"Census attr entry {key!r} must be exactly {{color, features}}")
        fields = {}
        for field in ("color", "features"):
            text_value = value[field]
            if not isinstance(text_value, str) or not text_value.strip() or len(text_value) > 200:
                raise ValueError(f"Census attr {key!r}.{field} is missing or oversized")
            fields[field] = text_value.strip()
        result[key] = fields
    return result


def trusted_objects(frame: dict) -> list[dict]:
    """Trusted object set of one completed frame: the single enumeration's list.

    The one definition of "which boxes of this frame may carry facts",
    consumed by the assembler, the reranker, the review session builder,
    and the adjustment report.
    """
    return frame["findall"]["objects"]


def select_frames(candidates: list[dict], k: int = 3) -> list[dict]:
    """Pick k frames: highest object count, ties broken by spread.

    Deterministic: primary key count descending, then greatest minimum
    distance to the already-chosen frames, then lowest frame number.
    """
    remaining = sorted(candidates, key=lambda c: (-c["count"], c["frame_no"]))
    chosen: list[dict] = []
    while remaining and len(chosen) < k:
        if not chosen:
            chosen.append(remaining.pop(0))
            continue
        best = max(
            remaining,
            key=lambda c: (
                c["count"],
                min(abs(c["frame_no"] - x["frame_no"]) for x in chosen),
                -c["frame_no"],
            ),
        )
        chosen.append(best)
        remaining.remove(best)
    return chosen


def findall_messages(marked_jpeg_url: str, *, prompt: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise visual-grounding enumerator. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "One complete RGB scene; the red rectangle marks the reference object."},
                {"type": "image_url", "image_url": {"url": marked_jpeg_url, "detail": "high"}},
                {"type": "text", "text": prompt or FINDALL_PROMPT},
            ],
        },
    ]


def attr_messages(numbered_jpeg_url: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise visual inspector. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "One complete RGB scene with numbered boxes around the enumerated objects."},
                {"type": "image_url", "image_url": {"url": numbered_jpeg_url, "detail": "high"}},
                {"type": "text", "text": ATTR_PROMPT},
            ],
        },
    ]

"""Census protocol: the teacher only reports facts, code validates everything.

Passes (v3 design, frozen in ``spec/census_protocol.md``, 2026-09-06):
- ``findall`` x2 independent passes per frame: enumerate up to 12 objects the
  teacher is most confident about (clear outline, nameable at a glance), the
  red-boxed category first when multiple instances exist, ordered left to
  right, each with its own category name and a normalized bbox. The red-boxed
  GT object must be included (it is the canary anchor, nothing more).
- ``attr``: numbered-box attribute report for the agreed object set of
  selected frames.

Deterministic gates in code (the teacher never self-certifies):
- canary: the returned set must re-find the GT box (IoU >= 0.5) — the only
  frame-fatal gate (the teacher pointing at the wrong target is unfixable)
- everything else is normalized in code, never fatal:
  ordering -> sorted by x1 and renumbered (v1 pilot: a fatal ordering gate
  killed 27/146 panoramic frames); near-identical boxes (IoU >= 0.95, the
  same object listed twice, e.g. nested trolley+robot listings) -> dedup;
  zero-area boxes -> dropped (degenerate under every convention)
- cross-pass agreement: one-to-one IoU >= 0.5 matching between the passes;
  the intersection is the trusted object set (the second pass IS the review)

Category naming may drift between passes/frames (swan/duck); normalization
is an assembler concern, not a census gate. Object identity across frames is
NOT established: every sample is fact-supported by its own frame only.

Prompts contain no example queries and no style options by design — the
teacher has nothing stylistic to imitate.
"""

from __future__ import annotations

import hashlib
import json
import re

from PIL import Image, ImageDraw

from foundry.bbox import compute_iou

FINDALL_PROMPT = """The red rectangle marks one object in the scene.
Task: list up to 12 objects you are MOST CONFIDENT about — clear outline, nameable at a glance — regardless of category, ordered from left to right. If multiple instances of the red-boxed category exist, include them all first, then fill the remaining slots with other confident objects. Skip tiny clutter, blurry ground debris, and anything you cannot identify precisely.
You must include the object inside the red rectangle. Number them 1..N (N is the total count).
For each object give a short common category name and its bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.
Output JSON only:
{"objects": [{"i": 1, "category": "<category name>", "bbox": [x1, y1, x2, y2]}, ...]}"""

ATTR_PROMPT = """The image shows numbered boxes around objects in the scene.
For each numbered object report only what is directly visible: its color and one notable visible feature.
Do not guess occluded or unclear properties.
Output JSON only:
{"1": {"color": "...", "features": "..."}, ...}"""

FINDALL_PROMPT_HASH = hashlib.sha256(FINDALL_PROMPT.encode("utf-8")).hexdigest()
ATTR_PROMPT_HASH = hashlib.sha256(ATTR_PROMPT.encode("utf-8")).hexdigest()

CANARY_IOU = 0.5
MATCH_IOU = 0.5
SELF_DUP_IOU = 0.95
BBOX_SLACK = 0.02
MAX_OBJECTS = 12


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


def _normalize_objects(cleaned: list[dict], gt_bbox: list[float]) -> list[dict]:
    """Sort by x1, drop near-identical duplicates, verify the canary.

    All three steps are invariant under the uniform scaling that separates
    the coordinate conventions, so the canary remains the sole convention
    discriminator.
    """
    cleaned.sort(key=lambda item: item["bbox"][0])
    kept: list[dict] = []
    for item in cleaned:
        if any(compute_iou(prev["bbox"], item["bbox"]) >= SELF_DUP_IOU for prev in kept):
            continue
        kept.append(item)
    if not any(compute_iou(item["bbox"], gt_bbox) >= CANARY_IOU for item in kept):
        raise ValueError("Census canary failed: red-boxed target not re-found")
    for index, item in enumerate(kept, start=1):
        item["i"] = index
    return kept


def parse_findall_response(
    text: object,
    *,
    gt_bbox: list[float],
    image_size: tuple[int, int] | None = None,
) -> dict:
    """Validate one findall response through every deterministic gate.

    The response is a panoramic enumeration: every entry carries its own
    category name. The teacher's coordinate convention is auto-detected: if
    all values are within [0, 1] the response is used directly; otherwise the
    per-mille (0-1000, GLM/Qwen family convention) and shown-image-pixel
    candidates are both normalized and the canary acts as the oracle — the
    first candidate whose set survives normalization (sort, dedup, degenerate
    drop) and passes the canary wins.
    """
    payload = _parse_json_object(text, label="Census findall")
    if set(payload) != {"objects"}:
        raise ValueError("Census findall schema must be exactly {objects}")
    objects = payload["objects"]
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError(f"Census findall must list 1..{MAX_OBJECTS} objects")
    raw_boxes: list[dict] = []
    for expected_i, item in enumerate(objects, start=1):
        if not isinstance(item, dict) or set(item) != {"i", "category", "bbox"}:
            raise ValueError("Census object entries must be exactly {i, category, bbox}")
        index = item["i"]
        if isinstance(index, bool) or not isinstance(index, int) or index != expected_i:
            raise ValueError("Census object indices must be sequential 1..N")
        category = item["category"]
        if not isinstance(category, str) or not category.strip() or len(category) > 64:
            raise ValueError(f"Census object {index} category is missing or oversized")
        raw_boxes.append({"i": index, "category": category.strip(), "bbox": item["bbox"]})

    flat = [value for item in raw_boxes for value in item["bbox"]]
    numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in flat
    )
    candidates: list[tuple[str, tuple[float, float]]] = []
    if numeric and all(-BBOX_SLACK <= value <= 1.0 + BBOX_SLACK for value in flat):
        candidates.append(("normalized-0-1", (1.0, 1.0)))
    else:
        candidates.append(("per-mille-0-1000", (1.0 / 1000.0, 1.0 / 1000.0)))
        if image_size is not None:
            candidates.append(("pixels-of-shown-image", (1.0 / image_size[0], 1.0 / image_size[1])))

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
        try:
            cleaned = _normalize_objects(cleaned, gt_bbox)
        except ValueError as exc:
            errors.append(f"[{name}] {exc}")
            continue
        return {"objects": cleaned, "bbox_convention": name}
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


def pass_agreement(objects_a: list[dict], objects_b: list[dict]) -> dict:
    """One-to-one greedy IoU matching between two independent findall passes."""
    pairs = []
    for a_index, a_item in enumerate(objects_a):
        for b_index, b_item in enumerate(objects_b):
            iou = compute_iou(a_item["bbox"], b_item["bbox"])
            if iou >= MATCH_IOU:
                pairs.append((iou, a_index, b_index))
    pairs.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched_boxes: list[dict] = []
    for iou, a_index, b_index in pairs:
        if a_index in used_a or b_index in used_b:
            continue
        used_a.add(a_index)
        used_b.add(b_index)
        matched_boxes.append(objects_a[a_index])
    count_a, count_b = len(objects_a), len(objects_b)
    union = count_a + count_b - len(matched_boxes)
    return {
        "matched": len(matched_boxes),
        "count_a": count_a,
        "count_b": count_b,
        "count_agree": count_a == count_b,
        "jaccard": len(matched_boxes) / union if union else 0.0,
        "agreed_objects": matched_boxes,
    }


def select_frames(candidates: list[dict], k: int = 3) -> list[dict]:
    """Pick k frames: highest agreed-object count, ties broken by spread.

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


def trusted_objects(frame: dict) -> list[dict]:
    """Trusted object set of one completed frame: the surviving pass's set
    when only one findall completed, else the two-pass intersection.

    The one definition of "which boxes of this frame may carry facts",
    consumed by the assembler, the reranker, the review session builder,
    and the adjustment report.
    """
    if frame.get("single_pass"):
        good = frame["findall_1"] if frame["findall_1"]["status"] == "completed" else frame["findall_2"]
        return good["objects"]
    return pass_agreement(frame["findall_1"]["objects"], frame["findall_2"]["objects"])["agreed_objects"]


def findall_messages(marked_jpeg_url: str, *, previous_error: str = "", prompt: str | None = None) -> list[dict]:
    repair = ""
    if previous_error:
        repair = (
            "\nThe previous response failed deterministic validation for this reason: "
            f"{previous_error}. Correct that failure and regenerate the complete JSON object."
        )
    return [
        {"role": "system", "content": "You are a precise visual-grounding enumerator. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "One complete RGB scene; the red rectangle marks the reference object."},
                {"type": "image_url", "image_url": {"url": marked_jpeg_url, "detail": "high"}},
                {"type": "text", "text": (prompt or FINDALL_PROMPT) + repair},
            ],
        },
    ]


def attr_messages(numbered_jpeg_url: str, *, previous_error: str = "") -> list[dict]:
    repair = ""
    if previous_error:
        repair = (
            "\nThe previous response failed deterministic validation for this reason: "
            f"{previous_error}. Correct that failure and regenerate the complete JSON object."
        )
    return [
        {"role": "system", "content": "You are a precise visual inspector. Return only valid JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "One complete RGB scene with numbered boxes around the enumerated objects."},
                {"type": "image_url", "image_url": {"url": numbered_jpeg_url, "detail": "high"}},
                {"type": "text", "text": ATTR_PROMPT + repair},
            ],
        },
    ]

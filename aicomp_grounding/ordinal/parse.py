"""Parse prompt assembly and strict decoding of the model's intent reply."""

from __future__ import annotations

import json
import re

from aicomp_grounding.ordinal import loader

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

#: Thinking block delimiters.  Qwen3.5 wraps its reasoning in these when
#: ``enable_thinking`` is on.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

#: Phrases that make the thinking text commit to a number.  Coordinates and
#: confidences are not matched.
_COUNT_CLAIM = re.compile(
    r"(?:there\s+(?:are|is)|i\s+(?:count|see|find|found)|(?:a\s+)?total\s+of|"
    r"count(?:ed)?|in\s+total)\s+(?:about\s+|around\s+|exactly\s+)?(\d{1,4})"
    r"|(\d{1,4})\s+(?:instances?|of\s+them|in\s+total|total|totals)",
    re.IGNORECASE,
)


def split_thinking(text: object) -> tuple[str | None, str]:
    """Split one reply into ``(thinking, answer)``.

    ``thinking`` is None when the reply carries no thinking block.  A block that
    was opened but never closed yields an empty answer.
    """
    if not isinstance(text, str):
        return None, ""
    start = text.find(THINK_OPEN)
    if start < 0:
        return None, text
    body = start + len(THINK_OPEN)
    end = text.find(THINK_CLOSE, body)
    if end < 0:
        return text[body:].strip(), ""
    thinking = text[body:end].strip()
    answer = (text[:start] + text[end + len(THINK_CLOSE):]).strip()
    return (thinking or None), answer


def thinking_reported_counts(text: object) -> set[int]:
    """Counts the thinking text explicitly claims; empty when it claims none."""
    if not isinstance(text, str) or not text.strip():
        return set()
    found: set[int] = set()
    for leading, trailing in _COUNT_CLAIM.findall(text):
        digits = leading or trailing
        if digits:
            found.add(int(digits))
    return found


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_object(text: object) -> dict | None:
    """Return the single JSON object in a model reply, or None.

    Accepts an optional ```json fence.  Rejects duplicate keys, arrays, prose
    around the object, and anything that is not exactly one JSON object.
    """
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    fence = _FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        payload = json.loads(candidate, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def build_parse_messages(query: str) -> list[dict]:
    """Text-only messages: the parse step never sees an image."""
    system, user = loader.load_prompt(loader.PARSE)
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": user.replace("{query}", query)}],
        },
    ]

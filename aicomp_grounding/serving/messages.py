"""Grounding message construction; the system prompt text lives in ``prompts/``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "grounding.md"

#: The system prompt ships as package data so the run identity can hash the file.
GROUNDING_SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

GROUNDING_PROMPT_PROTOCOL = {
    "version": 1,
    "modality_order": ["visible", "infrared", "depth"],
    "query_template": "Locate: {query}",
    "output_protocol": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>:integer_0_1000",
}


def grounding_prompt_hash(system_prompt: str) -> str:
    payload = {**GROUNDING_PROMPT_PROTOCOL, "system_prompt": system_prompt}
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_grounding_messages(visible_image, infrared_image, depth_image, query: str) -> list[dict]:
    """Build inference messages from modalities and query only; no label is accepted."""
    return [
        {"role": "system", "content": [{"type": "text", "text": GROUNDING_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": visible_image},
                {"type": "image", "image": infrared_image},
                {"type": "image", "image": depth_image},
                {"type": "text", "text": f"Locate: {query}"},
            ],
        },
    ]


def build_training_messages(
    visible_image,
    infrared_image,
    depth_image,
    query: str,
    bbox_text: str,
) -> list[dict]:
    """Build supervised messages using the same prompt prefix as inference."""
    messages = build_grounding_messages(visible_image, infrared_image, depth_image, query)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": bbox_text}]})
    return messages

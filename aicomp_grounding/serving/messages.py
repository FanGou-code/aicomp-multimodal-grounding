"""Grounding message construction; the prompt texts live in ``prompts/*.md``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aicomp_grounding.serving.prompt_files import read_prompt_text, split_prompt_pair

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

GROUNDING_PROMPT_PROTOCOL = {
    "version": 1,
    "modality_order": ["visible"],
    "output_protocol": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>:integer_0_1000",
}

GLM_GROUNDING_PROMPT_PROTOCOL = {
    "version": 1,
    "modality_order": ["visible"],
    "output_protocol": "<|begin_of_box|>x1,y1,x2,y2<|end_of_box|>:integer_0_1000",
    "enable_thinking": False,
}


def load_prompt_pair(name: str) -> tuple[str, str]:
    """Split ``prompts/<name>.md`` into its ``(system, user)`` parts."""
    label = f"Grounding prompt {name!r}"
    text = read_prompt_text(PROMPT_DIR / f"{name}.md", label=label)
    return split_prompt_pair(text, label=label)


GROUNDING_SYSTEM_PROMPT, GROUNDING_USER_TEMPLATE = load_prompt_pair("grounding")
GLM46V_SYSTEM_PROMPT, GLM46V_USER_TEMPLATE = load_prompt_pair("glm46v")


def grounding_prompt_hash(protocol: dict, system_prompt: str, user_template: str) -> str:
    """Hash one grounding prompt protocol together with its editable prompt texts."""
    payload = {
        **protocol,
        "system_prompt": system_prompt,
        "user_template": user_template,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_grounding_messages(visible_image, query: str) -> list[dict]:
    """Build inference messages from the visible image and query only; no label is accepted."""
    return [
        {"role": "system", "content": [{"type": "text", "text": GROUNDING_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": visible_image},
                {"type": "text", "text": GROUNDING_USER_TEMPLATE.format(query=query)},
            ],
        },
    ]


def build_training_messages(
    visible_image,
    query: str,
    bbox_text: str,
) -> list[dict]:
    """Build supervised messages using the same prompt prefix as inference."""
    messages = build_grounding_messages(visible_image, query)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": bbox_text}]})
    return messages

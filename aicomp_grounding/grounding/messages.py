"""Grounding message construction; prompt texts live in ``prompts/<model>.md``.

One prompt file per model so each model's protocol and wording are independently
editable. Prompt texts are files so prompt edits are text edits and the run
identity can hash everything that changes the model's behaviour. A missing,
empty, or malformed file is a hard error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

SYSTEM_MARK = "[system]"
USER_MARK = "[user]"

QWEN3VL_PROMPT_PROTOCOL = {
    "version": 1,
    "modality_order": ["visible"],
    "output_protocol": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>:integer_0_1000",
}

QWEN3_5_PROMPT_PROTOCOL = {
    "version": 1,
    "modality_order": ["visible"],
    "output_protocol": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>:integer_0_1000",
}

GLM46V_PROMPT_PROTOCOL = {
    "version": 1,
    "modality_order": ["visible"],
    "output_protocol": "<|begin_of_box|>x1,y1,x2,y2<|end_of_box|>:integer_0_1000",
    "enable_thinking": False,
}


def load_prompt_pair(name: str) -> tuple[str, str]:
    """Split ``prompts/<name>.md`` into its ``(system, user)`` parts."""
    label = f"Grounding prompt {name!r}"
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"{label} prompt file is missing: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} prompt file is empty: {path}")
    if not text.startswith(SYSTEM_MARK):
        raise ValueError(f"{label} prompt must start with {SYSTEM_MARK}")
    head, separator, tail = text.partition(USER_MARK)
    if not separator:
        raise ValueError(f"{label} prompt is missing the {USER_MARK} marker")
    system = head[len(SYSTEM_MARK) :].strip()
    user = tail.strip()
    if not system or not user:
        raise ValueError(f"{label} prompt has an empty system or user part")
    return system, user


QWEN3VL_SYSTEM_PROMPT, QWEN3VL_USER_TEMPLATE = load_prompt_pair("qwen3vl")
QWEN3_5_SYSTEM_PROMPT, QWEN3_5_USER_TEMPLATE = load_prompt_pair("qwen3_5")
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


def build_grounding_messages(
    visible_image,
    query: str,
    *,
    system_prompt: str,
    user_template: str,
) -> list[dict]:
    """Build inference messages from the visible image and query only; no label is accepted."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": visible_image},
                {"type": "text", "text": user_template.format(query=query)},
            ],
        },
    ]


def build_training_messages(
    visible_image,
    query: str,
    bbox_text: str,
    *,
    system_prompt: str,
    user_template: str,
) -> list[dict]:
    """Build supervised messages using the same prompt prefix as inference."""
    messages = build_grounding_messages(
        visible_image, query, system_prompt=system_prompt, user_template=user_template
    )
    messages.append({"role": "assistant", "content": [{"type": "text", "text": bbox_text}]})
    return messages

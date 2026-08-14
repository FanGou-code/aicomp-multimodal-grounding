"""Prompt construction kept separate so leakage can be tested offline."""

from __future__ import annotations

import hashlib
import json

GROUNDING_SYSTEM_PROMPT = (
    "You are a visual grounding assistant. The images are ordered as visible RGB, "
    "infrared, and depth. Locate the object described by the query. Output only "
    "<|box_start|>(x1,y1),(x2,y2)<|box_end|>, using integer coordinates from 0 to 1000."
)

RETRY_GROUNDING_SYSTEM_PROMPT = (
    "You are a visual grounding assistant. The images are ordered as visible RGB, "
    "infrared, and depth. Locate the object described by the query. "
    "You MUST output EXACTLY two coordinate pairs forming a bounding box: "
    "<|box_start|>(x1,y1),(x2,y2)<|box_end|> with integer coordinates from 0 to 1000. "
    "NEVER refuse, NEVER say the object is absent, NEVER explain. "
    "If uncertain, output your best-guess bounding box anyway. "
    "Output ONLY the box token, nothing else."
)

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


def build_retry_grounding_messages(visible_image, infrared_image, depth_image, query: str) -> list[dict]:
    """Build messages with the stronger retry prompt for previously failed queries."""
    return [
        {"role": "system", "content": [{"type": "text", "text": RETRY_GROUNDING_SYSTEM_PROMPT}]},
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


def standardize_query(query: str) -> str:
    """
    Standardize visual grounding query text for training and inference symmetry.

    Normalization rules:
    1. Strip leading/trailing whitespace
    2. Remove trailing punctuation (. ? ! : ;)
    3. Collapse multiple spaces/tabs into single space
    4. Normalize curly quotes to ASCII quotes
    5. Capitalize first letter (Sentence Case)

    Args:
        query: Raw query text

    Returns:
        Cleaned query text with consistent formatting
    """
    if not query:
        return ""

    # Strip whitespace
    q = query.strip()

    # Remove trailing punctuation
    q = q.rstrip(".?!:;").strip()

    # Collapse multiple spaces
    q = " ".join(q.split())

    # Normalize curly quotes to ASCII
    q = q.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")

    # Capitalize first letter
    if q and q[0].islower():
        q = q[0].upper() + q[1:]

    return q


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

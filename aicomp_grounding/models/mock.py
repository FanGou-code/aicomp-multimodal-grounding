"""Deterministic mock adapter: enables torch-free end-to-end pipeline tests.

Loads nothing and answers every query with a stable pseudo-random valid box
derived from the query text, so the full chain
(items -> adapter -> predictions -> evaluation -> submission ZIP)
runs in CI without a GPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from aicomp_grounding.models.base import ModelInput, Prediction


def _stable_box(query: str) -> list[float]:
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    x1 = 0.05 + (digest[0] / 255.0) * 0.40
    y1 = 0.05 + (digest[1] / 255.0) * 0.40
    x2 = x1 + 0.15 + (digest[2] / 255.0) * 0.25
    y2 = y1 + 0.15 + (digest[3] / 255.0) * 0.25
    return [round(x1, 6), round(y1, 6), round(min(x2, 1.0), 6), round(min(y2, 1.0), 6)]


_MOCK_INTENT = (
    '{"category": "object", "selection": {"mode": "rank", "k": 1, "axis": "x", "direction": "asc"}}'
)

_MOCK_ENUMERATION = (
    '{"count": 2, "instances": ['
    '{"bbox": [0.10, 0.10, 0.20, 0.30], "confidence": 0.9}, '
    '{"bbox": [0.40, 0.10, 0.50, 0.30], "confidence": 0.9}]}'
)


class MockAdapter:
    name = "mock"
    model_name = "mock/grounding"
    model_revision = "static"
    supports_lora = False

    def __init__(self):
        self.generation_config: dict[str, Any] = {"mock": True}

    def prompt_hash(self) -> str:
        return "mock-adapter-v1"

    def identity(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "prompt_hash": self.prompt_hash(),
        }

    def load(
        self,
        *,
        device: str = "cuda",
        lora_path: Path | None = None,
        model_path: str | None = None,
    ) -> None:
        if lora_path is not None:
            raise ValueError("MockAdapter carries no LoRA weights")

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        return [
            Prediction(bbox=_stable_box(sample.query), score=0.5)
            for sample in samples
        ]

    def generate_messages(
        self,
        messages_list: list[list[dict]],
        *,
        max_new_tokens: int,
        temperature: float,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Deterministic replies: an image means the enumerate prompt, none means parse.

        Both calls return the same list, so the reconcile gate passes and the
        ordinal chain stays exercisable end to end without a GPU.
        """
        replies: list[str] = []
        for messages in messages_list:
            has_image = any(
                part.get("type") == "image"
                for message in messages
                for part in message["content"]
            )
            replies.append(_MOCK_ENUMERATION if has_image else _MOCK_INTENT)
        return replies

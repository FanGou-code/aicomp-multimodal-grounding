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

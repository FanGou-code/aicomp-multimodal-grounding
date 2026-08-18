"""Shared contracts for per-model grounding adapters.

An adapter owns everything model-specific on the inference path:

- ``identity()``        fields consumed by the fingerprint system
- ``generation_config`` decoding parameters recorded in run metadata
- ``load()``            weights + processor (lazy heavy imports inside)
- ``predict()``         (three images, query) -> Prediction, in batches

Pure logic (prompt construction, output parsing, coordinate conversion) lives
in adapter modules as module-level functions so it stays unit-testable in
torch-free CI; only ``load``/``predict`` require a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL.Image import Image


@dataclass(frozen=True)
class Prediction:
    """Unified grounding output: normalized XYXY box plus optional score."""

    bbox: list[float] | None
    score: float | None = None


@dataclass
class ModelInput:
    """One query with its three aligned modality images (PIL, RGB)."""

    visible: Image
    infrared: Image
    depth: Image
    query: str
    key: str = field(default="")


class GroundingAdapter(Protocol):
    """Structural interface; adapters are plain classes, no inheritance needed."""

    name: str
    model_name: str
    model_revision: str
    #: Whether this adapter accepts a LoRA adapter directory on load().
    supports_lora: bool
    #: Decoding parameters recorded into the inference run identity.
    generation_config: dict[str, Any]

    def prompt_hash(self) -> str:
        """Stable hash of this model's inference prompt protocol."""
        ...

    def identity(self) -> dict[str, Any]:
        """Model identity fields consumed by fingerprinting."""
        ...

    def load(
        self,
        *,
        device: str = "cuda",
        lora_path: Path | None = None,
        model_path: str | None = None,
    ) -> None:
        """Load weights; heavy dependencies import lazily inside.

        ``model_path`` overrides the hub id with a local weights directory
        (offline machines); identity/fingerprinting always use the canonical
        ``model_name``/``model_revision``.
        """
        ...

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        """Predict a batch of tri-modal inputs; order is preserved."""
        ...

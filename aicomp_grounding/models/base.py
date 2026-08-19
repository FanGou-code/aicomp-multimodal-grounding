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


class TrainableGroundingAdapter(Protocol):
    """Training-side contract for adapters that can drive the generic loop."""

    name: str
    model_name: str
    model_revision: str
    supports_lora: bool

    def training_hyperparameters(self) -> dict[str, Any]:
        """Return the model-specific training hyperparameters."""
        ...

    def lora_target_modules(self) -> list[str]:
        """Return PEFT target module names for this model."""
        ...

    def load_for_training(
        self,
        *,
        device: str = "cuda",
        lora_path: Path | None = None,
        model_path: str | None = None,
    ) -> tuple[Any, Any]:
        """Load and return ``(model, processor)`` for training."""
        ...

    def build_training_batch(
        self,
        item: dict,
        *,
        data_root: Path,
        processor: Any,
    ) -> dict:
        """Build one model-ready training example with labels."""
        ...

    def collate_training_batch(
        self,
        batch: list[dict],
        *,
        processor: Any,
    ) -> dict:
        """Pad and stack model-specific training examples."""
        ...

    def build_grounding_batch(
        self,
        samples: list[ModelInput],
        *,
        processor: Any,
    ) -> dict:
        """Build model-specific generation inputs for a batch."""
        ...

    def decode_grounding_outputs(
        self,
        processor: Any,
        generated_ids: Any,
        prompt_len: int,
    ) -> list[str]:
        """Decode generated grounding text outputs."""
        ...

    def parse_grounding_text(self, text: str) -> list[float] | None:
        """Parse a generated grounding text into normalized XYXY."""
        ...

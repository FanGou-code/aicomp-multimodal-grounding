"""Shared contracts for per-model grounding adapters.

An adapter owns everything model-specific on the inference path:

- ``identity()``        fields consumed by the fingerprint system
- ``generation_config`` decoding parameters recorded in run metadata
- ``load()``            weights + processor (lazy heavy imports inside)
- ``predict()``         (visible image, query) -> Prediction, in batches

Pure logic (prompt construction, output parsing, coordinate conversion) lives
in adapter modules as module-level functions so it stays unit-testable in
torch-free CI; only ``load``/``predict`` require a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL.Image import Image


#: Language-model projection names for adapters whose model exposes the standard
#: split MLP.  An adapter whose model fuses or renames projections declares its
#: own set rather than inheriting this one.
DEFAULT_LORA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def language_model_lora_targets(*projections: str) -> str:
    """Build this adapter's PEFT target regex from its own projection names.

    The anchoring is the load-bearing part, and it is why this is a shared
    builder rather than a shared value: which projections a model exposes is a
    per-architecture fact, but *where* they live is the recipe.

    PEFT matches a list of target names by module suffix, so a bare
    ``["q_proj", ...]`` list also matches any vision-tower projection that reuses
    a language-model name -- the Qwen2.5-VL-style vision MLP (``gate_proj`` /
    ``up_proj`` / ``down_proj``) and the InternViT attention (``q_proj`` /
    ``k_proj`` / ``v_proj``).  A trainable vision tower is not part of this
    recipe: it puts the vision encoder through backward plus gradient-checkpoint
    recomputation and leaves adapters mutually incomparable.  PEFT matches a
    *string* ``target_modules`` with ``re.fullmatch``, so the
    ``model.language_model`` prefix and the trailing ``$`` keep the encoder
    frozen whichever names an adapter declares.
    """
    alternation = "|".join(re.escape(name) for name in projections)
    return rf"model\.language_model\..*\.(?:{alternation})$"


def require_local_model_path(model_path: str | Path | None) -> str:
    """Require a pre-downloaded model; model execution never downloads files."""
    if model_path is None or not str(model_path).strip():
        raise ValueError(
            "Automatic model download is disabled. Download the model first "
            "and pass --model-path pointing to its local directory."
        )
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Local model directory is missing config.json: {path}")
    return str(path)


def validated_prompt_length(full_input_ids, prompt_input_ids, *, sample_id: str) -> int:
    if len(full_input_ids.shape) != 2 or len(prompt_input_ids.shape) != 2:
        raise ValueError(f"Training token IDs must be rank 2 for {sample_id}")
    prompt_length = int(prompt_input_ids.shape[1])
    if (
        full_input_ids.shape[0] != prompt_input_ids.shape[0]
        or prompt_length <= 0
        or prompt_length >= full_input_ids.shape[1]
    ):
        raise ValueError(f"Training label boundary is invalid for {sample_id}")
    if not bool((full_input_ids[:, :prompt_length] == prompt_input_ids).all().item()):
        raise ValueError(f"Training prompt tokens are not an exact prefix for {sample_id}")
    return prompt_length


@dataclass(frozen=True)
class Prediction:
    """Unified grounding output: normalized XYXY box."""

    bbox: list[float] | None


@dataclass
class ModelInput:
    """One query with its visible RGB image (PIL)."""

    visible: Image
    query: str
    key: str = field(default="")


class GroundingAdapter(Protocol):
    """Structural interface; adapters are plain classes, no inheritance needed."""

    name: str
    model_name: str
    model_revision: str
    compute_dtype: str
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

        ``model_path`` is the required pre-downloaded weights directory;
        identity/fingerprinting always use the canonical
        ``model_name``/``model_revision``.
        """
        ...

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        """Predict a batch of (visible image, query) inputs; order is preserved."""
        ...

    #: Adapters that split preprocessing from generation set this True so the
    #: DataLoader can run ``prepare_inputs`` inside worker processes.
    supports_prepared_inputs: bool

    def prepare_inputs(self, samples: list[ModelInput]) -> dict:
        """CPU-only batch preprocessing (prompt + processor tensors).

        Must not touch the loaded model.  Runs in DataLoader workers when
        ``supports_prepared_inputs`` is True, overlapping image preprocessing
        with GPU generation.
        """
        ...

    def predict_from_inputs(self, inputs: dict) -> list[Prediction]:
        """GPU-side generation from ``prepare_inputs`` output; order kept."""
        ...

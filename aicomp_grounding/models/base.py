"""Shared contracts for per-model grounding adapters.

An adapter owns everything model-specific on the inference path:

- ``identity()``        fields consumed by the fingerprint system
- ``generation_config`` decoding parameters recorded in run metadata
- ``load()``            weights + processor (lazy heavy imports inside)
- ``predict()``         (three images, query) -> Prediction, in batches
- ``generate_messages()`` caller-built chat messages -> raw text, for callers
                        that own their prompt and their output shape

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


def run_generation(
    model: Any,
    processor: Any,
    inputs: dict,
    *,
    max_new_tokens: int,
    temperature: float,
    skip_special_tokens: bool = False,
) -> list[str]:
    """Generate raw text from processor inputs; one string per sample.

    ``temperature <= 0`` decodes greedily, and the temperature argument is only
    passed on the sampled path so a greedy call never depends on an ignored
    parameter.  Shared by the adapters' ``generate_messages`` so the slicing and
    decoding rules have exactly one home.
    """
    import torch

    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, **kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = generated_ids[:, prompt_len:]
    return processor.batch_decode(
        generated_tokens, skip_special_tokens=skip_special_tokens
    )


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

        ``model_path`` is the required pre-downloaded weights directory;
        identity/fingerprinting always use the canonical
        ``model_name``/``model_revision``.
        """
        ...

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        """Predict a batch of tri-modal inputs; order is preserved."""
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


    def generate_messages(
        self,
        messages_list: list[list[dict]],
        *,
        max_new_tokens: int,
        temperature: float,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Template and generate caller-built chat messages; order preserved.

        Unlike ``predict``, the caller owns the prompt and the output shape:
        this returns raw decoded text.  The ordinal module needs it to enumerate
        with a different prompt, a single image, and sampled decoding.
        """
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

    def lora_target_modules(self) -> str:
        """Return this adapter's PEFT target regex; build it with
        :func:`language_model_lora_targets` so the vision tower stays frozen."""
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

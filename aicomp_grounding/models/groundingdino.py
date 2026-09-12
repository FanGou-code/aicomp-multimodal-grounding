"""GroundingDINO-B grounding adapter (zero-shot, RGB-only).

GroundingDINO is a discriminative detector: it consumes exactly ONE image and
a bare text query (no prompt template) and natively emits confidence scores,
which is exactly what WBF fusion needs. Infrared and depth are accepted by
the interface but ignored by design — see docs/handoff.md for the
rationale (single-image architecture; pseudo-color/thermal are
out-of-distribution for its Swin encoder).

Pure post-process selection helper is unit-tested; the GPU path uses the
explicit transformers ``GroundingDinoForObjectDetection`` class and has not
yet been smoke-tested (do a val slice first).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.models.base import ModelInput, Prediction, require_local_model_path

MODEL_NAME = "IDEA-Research/grounding-dino-base"
MODEL_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.30


def normalize_grounding_query(query: str) -> str:
    """Normalize a GroundingDINO caption the way the official demo does."""
    text = query.strip().lower()
    if not text.endswith("."):
        text += "."
    return text


def select_top_detection(
    boxes_xyxy: list[list[float]],
    scores: list[float],
    *,
    box_threshold: float = BOX_THRESHOLD,
) -> tuple[list[float] | None, float | None]:
    """Pick the highest-confidence valid post-processed XYXY box."""
    if not boxes_xyxy or not scores:
        return None, None
    for index in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
        if scores[index] < box_threshold:
            break
        # Float regression can push boxes a hair outside [0, 1]; clip before
        # validation so edge-of-frame targets are not wrongly discarded.
        clipped = [min(1.0, max(0.0, value)) for value in boxes_xyxy[index]]
        xyxy = validate_bbox(clipped)
        if xyxy is not None:
            return xyxy, float(scores[index])
    return None, None


def _load_local_processor(source: str):
    """Load a GroundingDINO processor from a local snapshot directory.

    ModelScope snapshots sometimes omit ``preprocessor_config.json``, which
    makes ``AutoProcessor.from_pretrained`` fail even though the tokenizer and
    model files are present. Rebuild the processor from its standard parts.
    """
    from transformers import (
        AutoTokenizer,
        GroundingDinoImageProcessor,
        GroundingDinoProcessor,
    )

    if (Path(source) / "preprocessor_config.json").is_file():
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(source, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    return GroundingDinoProcessor(
        image_processor=GroundingDinoImageProcessor(),
        tokenizer=tokenizer,
    )


class GroundingDINOAdapter:
    name = "groundingdino"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    supports_lora = False
    compute_dtype = "float32"

    def __init__(self, *, box_threshold: float = BOX_THRESHOLD):
        self.box_threshold = box_threshold
        self.generation_config: dict[str, Any] = {
            "box_threshold": box_threshold,
            "text_threshold": TEXT_THRESHOLD,
        }
        self._processor = None
        self._model = None

    def prompt_hash(self) -> str:
        # GroundingDINO consumes the bare query: no template to hash.
        return "groundingdino-bare-query-v1"

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
        source = require_local_model_path(model_path)

        import torch

        if lora_path is not None:
            raise ValueError("GroundingDINO path is zero-shot only; no LoRA support")

        processor = _load_local_processor(source)
        try:
            from transformers import GroundingDinoForObjectDetection

            model_class = GroundingDinoForObjectDetection
        except ImportError:
            from transformers import AutoModelForObjectDetection

            model_class = AutoModelForObjectDetection

        try:
            model = model_class.from_pretrained(
                source,
                local_files_only=True,
                dtype=torch.float32,
            )
        except TypeError:
            model = model_class.from_pretrained(
                source,
                local_files_only=True,
                torch_dtype=torch.float32,
            )
        model = model.to(device)
        model.eval()
        self._processor = processor
        self._model = model

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("GroundingDINOAdapter.load() must run before predict()")

        predictions: list[Prediction] = []
        for sample in samples:
            inputs = self._processor(
                images=sample.visible,
                text=normalize_grounding_query(sample.query),
                return_tensors="pt",
            ).to(self._model.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # post_process converts normalized cxcywh into normalized XYXY.
            post_process = self._processor.post_process_grounded_object_detection
            threshold_kw = (
                "threshold"
                if "threshold" in inspect.signature(post_process).parameters
                else "box_threshold"
            )
            results = post_process(
                outputs,
                inputs.input_ids,
                **{threshold_kw: self.box_threshold},
                text_threshold=TEXT_THRESHOLD,
            )[0]
            bbox, score = select_top_detection(
                results["boxes"].tolist(),
                results["scores"].tolist(),
                box_threshold=self.box_threshold,
            )
            predictions.append(Prediction(bbox=bbox, score=score))
        return predictions

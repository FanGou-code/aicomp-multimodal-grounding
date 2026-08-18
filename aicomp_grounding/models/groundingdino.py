"""GroundingDINO-B grounding adapter (zero-shot, RGB-only).

GroundingDINO is a discriminative detector: it consumes exactly ONE image and
a bare text query (no prompt template) and natively emits confidence scores,
which is exactly what WBF fusion needs. Infrared and depth are accepted by
the interface but ignored by design — see docs/handoff.md for the
rationale (single-image architecture; pseudo-color/thermal are
out-of-distribution for its Swin encoder).

Pure coordinate helpers are unit-tested; the GPU path uses the plain
transformers ``AutoModelForObjectDetection`` API and has not yet been
smoke-tested (do a val slice first).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.models.base import ModelInput, Prediction

MODEL_NAME = "IDEA-Research/grounding-dino-base"
# TODO(teammate): pin the exact commit hash before any recorded run.
MODEL_REVISION = "main"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.30


def cxcywh_to_xyxy(box: list[float]) -> list[float]:
    """Convert a normalized center-size box to normalized XYXY."""
    cx, cy, w, h = box
    return [
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ]


def select_top_detection(
    boxes_cxcywh: list[list[float]],
    scores: list[float],
) -> tuple[list[float] | None, float | None]:
    """Pick the highest-confidence valid box; (None, None) when nothing passes."""
    if not boxes_cxcywh or not scores:
        return None, None
    best_index = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_index] < BOX_THRESHOLD:
        return None, None
    xyxy = validate_bbox(cxcywh_to_xyxy(boxes_cxcywh[best_index]))
    if xyxy is None:
        return None, None
    return xyxy, float(scores[best_index])


class GroundingDINOAdapter:
    name = "groundingdino"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    supports_lora = False

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
        import torch
        from transformers import AutoModelForObjectDetection, AutoProcessor

        if lora_path is not None:
            raise ValueError("GroundingDINO path is zero-shot only; no LoRA support")

        source = model_path or self.model_name
        from_hub = model_path is None
        processor = AutoProcessor.from_pretrained(
            source, **({"revision": self.model_revision} if from_hub else {})
        )
        model = AutoModelForObjectDetection.from_pretrained(
            source,
            **({"revision": self.model_revision} if from_hub else {}),
            # Swin detectors are float32-trained; fp16 here is not worth the risk.
            torch_dtype=torch.float32,
        ).to(device)
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
                text=sample.query,
                return_tensors="pt",
            ).to(self._model.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Without target_sizes the processor keeps boxes as normalized
            # cxcywh, which is exactly what select_top_detection expects.
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=TEXT_THRESHOLD,
            )[0]
            bbox, score = select_top_detection(
                results["boxes"].tolist(),
                results["scores"].tolist(),
            )
            predictions.append(Prediction(bbox=bbox, score=score))
        return predictions

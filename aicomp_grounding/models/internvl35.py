"""InternVL3.5-8B grounding adapter (zero-shot skeleton for contributors).

Verification status
-------------------
Pure logic (question construction, box parsing, 0-1000 scaling) is pinned by
unit tests. The GPU path follows the official ``OpenGVLab/InternVL3_5-8B-HF``
transformers-native usage but has NOT been smoke-tested on a GPU yet:
whoever trains/evaluates this direction must first run a small val slice and
cross-check coordinates against the official ``evaluate_grounding.py`` script
(InternVL has a documented history of axis-order pitfalls; the parser below
pins [x1, y1, x2, y2] on a 0-1000 grid).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.models.base import ModelInput, Prediction

MODEL_NAME = "OpenGVLab/InternVL3_5-8B-HF"
MODEL_REVISION = "741a7d03020411e666c6109218ab71e08151ef86"

MAX_NEW_TOKENS = 64

_NUMBER = r"\d+(?:\.\d+)?"
_BOX_PATTERN = re.compile(
    rf"<box>\s*\[\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*\]\s*</box>"
)
_BARE_PATTERN = re.compile(
    rf"\[\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*\]"
)

MODALITY_PREAMBLE = (
    "The images are ordered as visible RGB, infrared, and depth. "
    "Image-1 is the visible RGB, Image-2 is the infrared, Image-3 is the depth. "
)
# Official grounding prompt from the InternVL team (see their evaluate_grounding
# script); do not modify without re-validating against reference outputs.
GROUNDING_PROMPT_SUFFIX = (
    "Please provide the bounding box coordinate of the region this sentence "
    "describes: <ref>{query}</ref>"
)


def build_grounding_question(query: str) -> str:
    """Full user question: three numbered <image> slots + official prompt."""
    return (
        "Image-1: <image>\nImage-2: <image>\nImage-3: <image>\n"
        + MODALITY_PREAMBLE
        + GROUNDING_PROMPT_SUFFIX.format(query=query)
    )


def parse_internvl_box(text: str) -> list[float] | None:
    """Parse ``<box>[[x1,y1,x2,y2]]</box>`` (0-1000) into normalized XYXY."""
    if not isinstance(text, str):
        return None
    match = _BOX_PATTERN.search(text)
    if match is None:
        # Some checkpoints emit the bare double-bracket list without tags.
        candidates = _BARE_PATTERN.findall(text)
        if len(candidates) != 1:
            return None
        match_values = candidates[0]
    else:
        match_values = match.groups()
    try:
        values = [float(value) / 1000.0 for value in match_values]
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= value <= 1.0 for value in values):
        return None
    return validate_bbox(values)


class InternVL35Adapter:
    name = "internvl35"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    supports_lora = True

    def __init__(self, *, max_num_tiles: int = 12):
        self.max_num_tiles = max_num_tiles
        self.generation_config: dict[str, Any] = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "max_num_tiles": max_num_tiles,
        }
        self._processor = None
        self._model = None

    def prompt_hash(self) -> str:
        import hashlib

        payload = {
            "modality_preamble": MODALITY_PREAMBLE,
            "grounding_prompt": GROUNDING_PROMPT_SUFFIX,
            "image_slots": ["<image>"] * 3,
        }
        import json

        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

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
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq, AutoProcessor

        source = model_path or self.model_name
        from_hub = model_path is None
        processor = AutoProcessor.from_pretrained(
            source,
            **({"revision": self.model_revision} if from_hub else {}),
            # Dynamic tiling budget per image (12 tiles ~ 1080p-class input).
            max_num_tiles=self.max_num_tiles,
        )
        processor.tokenizer.padding_side = "left"

        model = AutoModelForVision2Seq.from_pretrained(
            source,
            **({"revision": self.model_revision} if from_hub else {}),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)

        if lora_path is not None:
            model = PeftModel.from_pretrained(model, str(lora_path)).to(device)

        model.eval()
        self._processor = processor
        self._model = model

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("InternVL35Adapter.load() must run before predict()")

        processor = self._processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": build_grounding_question(sample.query)},
                ],
            }
            for sample in samples
        ]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        images = [
            image
            for sample in samples
            for image in (sample.visible, sample.infrared, sample.depth)
        ]
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, prompt_len:]
        text_outputs = processor.batch_decode(
            generated_tokens, skip_special_tokens=False
        )
        return [
            Prediction(bbox=parse_internvl_box(text), score=None)
            for text in text_outputs
        ]

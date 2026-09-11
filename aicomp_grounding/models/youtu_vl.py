"""Youtu-VL-4B-Instruct grounding adapter (inference-only, Modal-resident).

Youtu-VL-4B is a 3B-class MLLM from Tencent Youtu Lab that emits bounding
boxes as absolute-pixel coordinate tokens (``<x_N>``/``<y_N>``, N in 0..2047)
wrapped in ``<ref>...</ref><box>...</box>``. Its REC/counting numbers lead
the 4B class, so it earns a WBF seat as a zero-shot member.

The adapter is inference-only: Youtu requires ``transformers>=4.56,<=4.57.1``
plus ``trust_remote_code`` (a custom ``youtu_vl`` architecture), which is
incompatible with the repo's pinned 5.15.1. It therefore runs in a dedicated
Modal Image; the adapter code itself stays import-clean so the rest of the
repo never loads the remote-code path. The parser is pure logic and is
unit-tested locally; the GPU path (image-input contract, multi-image
``img_input`` kwarg) is verified on Modal before a recorded run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.models.base import ModelInput, Prediction

MODEL_NAME = "tencent/Youtu-VL-4B-Instruct"
MODEL_REVISION = "8d30a0e49662a1d628a472b12df264dbcd768753"

MAX_NEW_TOKENS = 64
REPETITION_PENALTY = 1.05

YOUTU_GROUNDING_PROMPT = (
    "The images are visible RGB, infrared, and depth of the same scene. "
    "Please provide the bounding box coordinate of the region this sentence "
    "describes: {query}"
)

_YOUTU_BOX = re.compile(r"<box>(.*?)</box>", re.DOTALL)
_YOUTU_COORD = re.compile(r"<[xy]_(\d+)>")


def parse_youtu_box(text: str, width: int, height: int) -> list[float] | None:
    """Parse Youtu ``<box><x_N><y_N><x_N><y_N></box>`` (absolute px) into XYXY.

    Coordinates are absolute pixels in the input image space (per the
    technical report); normalize by the image width/height the model saw.
    When multiple boxes are emitted, keep the largest-area valid one.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    if width <= 0 or height <= 0:
        return None
    best = None
    best_area = -1.0
    for box_text in _YOUTU_BOX.findall(text):
        nums = _YOUTU_COORD.findall(box_text)
        if len(nums) != 4:
            continue
        x1, y1, x2, y2 = (int(n) for n in nums)
        if x1 >= x2 or y1 >= y2:
            continue
        vals = [x1 / width, y1 / height, x2 / width, y2 / height]
        clipped = [min(1.0, max(0.0, v)) for v in vals]
        bbox = validate_bbox(clipped)
        if bbox is None:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > best_area:
            best_area = area
            best = bbox
    return best


def _build_messages(visible, infrared, depth, query: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": visible},
                {"type": "image", "image": infrared},
                {"type": "image", "image": depth},
                {"type": "text", "text": YOUTU_GROUNDING_PROMPT.format(query=query)},
            ],
        }
    ]


class YoutuVLAdapter:
    name = "youtu_vl"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    supports_lora = False

    def __init__(self):
        self.generation_config: dict[str, Any] = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "repetition_penalty": REPETITION_PENALTY,
        }
        self._processor = None
        self._model = None

    def prompt_hash(self) -> str:
        import hashlib
        import json

        payload = {
            "grounding_prompt": YOUTU_GROUNDING_PROMPT,
            "image_count": 3,
            "box_format": "<box><x_N><y_N><x_N><y_N></box>",
        }
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
        from transformers import AutoModelForCausalLM, AutoProcessor

        if lora_path is not None:
            raise ValueError("Youtu-VL is zero-shot only; no LoRA support")

        source = model_path or self.model_name
        from_hub = model_path is None
        revision_kwargs = {"revision": self.model_revision} if from_hub else {}
        processor = AutoProcessor.from_pretrained(
            source,
            trust_remote_code=True,
            use_fast=True,
            **revision_kwargs,
        )
        processor.tokenizer.padding_side = "left"
        print(
            f"[youtu_vl] image processor: {type(processor.image_processor).__name__}",
            flush=True,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                source,
                trust_remote_code=True,
                **revision_kwargs,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                source,
                trust_remote_code=True,
                **revision_kwargs,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        model = model.to(device)
        model.eval()
        self._processor = processor
        self._model = model

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("YoutuVLAdapter.load() must run before predict()")

        processor = self._processor
        predictions: list[Prediction] = []
        for sample in samples:
            images = [sample.visible, sample.infrared, sample.depth]
            width, height = sample.visible.size
            messages = _build_messages(
                sample.visible, sample.infrared, sample.depth, sample.query
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                generated_ids = self._model.generate(
                    **inputs,
                    img_input=images,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=REPETITION_PENALTY,
                )
            prompt_len = inputs["input_ids"].shape[1]
            text = processor.batch_decode(
                generated_ids[:, prompt_len:], skip_special_tokens=False
            )[0]
            bbox = parse_youtu_box(text, width, height)
            predictions.append(Prediction(bbox=bbox, score=None))
        return predictions

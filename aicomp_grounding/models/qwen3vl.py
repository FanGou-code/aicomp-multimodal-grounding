"""Qwen3-VL grounding adapter: the reference implementation.

Historical constants (model id / revision / pixel budgets) moved here from
``aicomp_grounding.config`` with unchanged values so every existing training
and inference fingerprint stays byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aicomp_grounding.bbox import parse_bbox_from_text
from aicomp_grounding.models.base import ModelInput, Prediction
from aicomp_grounding.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    build_grounding_messages,
    grounding_prompt_hash,
)

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

# Pixel budgets in Qwen processor units (28x28 per patch); 3072 patches
# covers a lossless 1920x1080 frame at ~2645 patches.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 3072 * 28 * 28

MAX_NEW_TOKENS = 32


class Qwen3VLAdapter:
    name = "qwen3vl"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    supports_lora = True

    def __init__(self, *, max_pixels: int = MAX_PIXELS):
        self.max_pixels = max_pixels
        self.generation_config: dict[str, Any] = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "max_pixels": max_pixels,
        }
        self._processor = None
        self._model = None

    def prompt_hash(self) -> str:
        return grounding_prompt_hash(GROUNDING_SYSTEM_PROMPT)

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
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        source = model_path or self.model_name
        from_hub = model_path is None
        processor = AutoProcessor.from_pretrained(
            source,
            **({"revision": self.model_revision} if from_hub else {}),
            min_pixels=MIN_PIXELS,
            max_pixels=self.max_pixels,
        )
        # Left padding keeps batched generation aligned for parsing.
        processor.tokenizer.padding_side = "left"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
        from qwen_vl_utils import process_vision_info

        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen3VLAdapter.load() must run before predict()")

        processor = self._processor
        messages_list = [
            build_grounding_messages(
                sample.visible, sample.infrared, sample.depth, sample.query
            )
            for sample in samples
        ]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
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
            Prediction(bbox=parse_bbox_from_text(text), score=None)
            for text in text_outputs
        ]

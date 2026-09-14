"""MiMo-VL-7B-RL grounding adapter.

MiMo-VL-7B-RL is Xiaomi's vision-language model fine-tuned with reinforcement
learning for multimodal grounding. It uses the Qwen2.5-VL architecture
(model_type ``qwen2_5_vl``) and outputs JSON with normalized bbox coordinates.

The model emits ``{"bbox_2d": [x1, y1, x2, y2]}`` where coordinates are
normalized 0-1. The adapter parses this JSON format and validates the bbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.config import RUNTIME_PYTHON_VERSION
from aicomp_grounding.models.base import (
    DEFAULT_LORA_PROJECTIONS,
    ModelInput,
    Prediction,
    language_model_lora_targets,
    require_local_model_path,
)
from aicomp_grounding.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    build_grounding_messages,
    build_training_messages,
    grounding_prompt_hash,
)
from aicomp_grounding.training_state import validated_prompt_length

MODEL_NAME = "XiaomiMiMo/MiMo-VL-7B-RL"
# ModelScope snapshot; the SOP download command pins this same revision. The
# adapter only reads the explicit local --model-path.
MODEL_REVISION = "d307865d4a3b6ad9ae35e574bcabaa563038c8fb"

# Pixel budgets in Qwen processor units (28x28 per patch); 3072 patches
# covers a lossless 1920x1080 frame at ~2645 patches.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 3072 * 28 * 28

MAX_NEW_TOKENS = 64


def _apply_chat_template(
    processor: Any,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _resolve_model_source(model_path: str | None) -> str:
    """Resolve the required pre-downloaded model directory."""
    return require_local_model_path(model_path)


def _parse_mimo_bbox(text: str) -> list[float] | None:
    """Parse MiMo-VL JSON bbox format: {"bbox_2d": [x1, y1, x2, y2]}."""
    import json
    import re

    # Extract JSON object from text
    match = re.search(r'\{[^}]*"bbox_2d"[^}]*\}', text)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        bbox = data.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        coords = [float(c) for c in bbox]
        if not validate_bbox(coords):
            return None
        return coords
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _format_mimo_bbox(bbox: list[float]) -> str:
    """Format bbox as MiMo-VL expects: {"bbox_2d": [x1, y1, x2, y2]}."""
    import json
    return json.dumps({"bbox_2d": bbox}, ensure_ascii=False)


class MiMoVLAdapter:
    name = "mimo_vl"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    supports_lora = True
    supports_prepared_inputs = True

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
        source = _resolve_model_source(model_path)

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=True,
            min_pixels=MIN_PIXELS,
            max_pixels=self.max_pixels,
        )
        # Left padding keeps batched generation aligned for parsing.
        processor.tokenizer.padding_side = "left"
        print(
            f"[mimo_vl] image processor: {type(processor.image_processor).__name__}",
            flush=True,
        )

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        model = model.to(device)

        if lora_path is not None:
            model = PeftModel.from_pretrained(model, str(lora_path), local_files_only=True).to(device)

        model.eval()
        self._processor = processor
        self._model = model

    def training_hyperparameters(self) -> dict[str, Any]:
        return {
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 1e-4,
            "epochs": 3,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "compute_dtype": "bfloat16",
            "autocast": True,
            "eval_batch_size": 4,
            "best_epoch_primary_metric": "acc_at_0_5",
            "python_version": RUNTIME_PYTHON_VERSION,
            "lora_targets": self.lora_target_modules(),
            "min_pixels": MIN_PIXELS,
            "max_pixels": self.max_pixels,
        }

    def lora_target_modules(self) -> str:
        return language_model_lora_targets(*DEFAULT_LORA_PROJECTIONS)

    def load_for_training(
        self,
        *,
        device: str = "cuda",
        lora_path: Path | None = None,
        model_path: str | None = None,
    ):
        self.load(device=device, lora_path=lora_path, model_path=model_path)
        # Training backward has no use for the KV cache and transformers would
        # force it off at forward time with a warning.  Setting it here keeps
        # the inference path (which does want the cache) untouched.
        self._model.config.use_cache = False
        return self._model, self._processor

    def build_training_batch(
        self,
        item: dict,
        *,
        data_root: Path,
        processor,
    ) -> dict:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        images = []
        for field in ("visible", "infrared", "depth"):
            with Image.open(data_root / item[field]) as opened:
                images.append(opened.convert("RGB"))
        visible, infrared, depth = images
        bbox_text = _format_mimo_bbox(item["bbox"])
        prompt_messages = build_grounding_messages(
            visible, infrared, depth, item["query"]
        )
        messages = build_training_messages(
            visible, infrared, depth, item["query"], bbox_text
        )
        text = _apply_chat_template(
            processor,
            messages,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        if "mm_token_type_ids" not in inputs:
            raise ValueError(
                f"MiMo-VL training batch is missing mm_token_type_ids for {item.get('key')}"
            )
        prompt_text = _apply_chat_template(
            processor,
            prompt_messages,
            add_generation_prompt=True,
        )
        prompt_inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        prompt_length = validated_prompt_length(
            inputs["input_ids"],
            prompt_inputs["input_ids"],
            sample_id=str(item.get("key", "")),
        )
        labels = inputs["input_ids"].clone()
        labels[0, :prompt_length] = -100
        result = {key: value.squeeze(0) for key, value in inputs.items()}
        result["labels"] = labels.squeeze(0)
        return result

    def collate_training_batch(self, batch: list[dict], *, processor) -> dict:
        import torch

        max_length = max(item["input_ids"].size(0) for item in batch)
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Processor tokenizer has no pad_token_id")
        ids, labels, attention, pixels, grids, mm_token_types = [], [], [], [], [], []
        for item in batch:
            if "mm_token_type_ids" not in item:
                raise ValueError("MiMo-VL training batch is missing mm_token_type_ids")
            padding = max_length - item["input_ids"].size(0)
            ids.append(
                torch.cat(
                    [item["input_ids"], torch.full((padding,), pad_id, dtype=torch.long)]
                )
            )
            labels.append(
                torch.cat(
                    [item["labels"], torch.full((padding,), -100, dtype=torch.long)]
                )
            )
            attention.append(
                torch.cat(
                    [item["attention_mask"], torch.zeros(padding, dtype=torch.long)]
                )
            )
            mm_token_types.append(
                torch.cat(
                    [
                        item["mm_token_type_ids"],
                        torch.zeros(padding, dtype=item["mm_token_type_ids"].dtype),
                    ]
                )
            )
            pixels.append(item["pixel_values"])
            grids.append(item["image_grid_thw"])
        return {
            "input_ids": torch.stack(ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention),
            "pixel_values": torch.cat(pixels, dim=0),
            "image_grid_thw": torch.cat(grids, dim=0),
            "mm_token_type_ids": torch.stack(mm_token_types),
        }

    def build_grounding_batch(self, samples: list[ModelInput], *, processor) -> dict:
        from qwen_vl_utils import process_vision_info

        messages = [
            build_grounding_messages(
                sample.visible,
                sample.infrared,
                sample.depth,
                sample.query,
            )
            for sample in samples
        ]
        texts = [
            _apply_chat_template(
                processor,
                message,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def decode_grounding_outputs(self, processor, generated_ids, prompt_len: int) -> list[str]:
        return processor.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=False,
        )

    def parse_grounding_text(self, text: str) -> list[float] | None:
        return _parse_mimo_bbox(text)

    def prepare_inputs(self, samples: list[ModelInput]) -> dict:
        from qwen_vl_utils import process_vision_info

        if self._processor is None:
            raise RuntimeError("MiMoVLAdapter.load() must run before predict()")

        processor = self._processor
        messages_list = [
            build_grounding_messages(
                sample.visible, sample.infrared, sample.depth, sample.query
            )
            for sample in samples
        ]
        texts = [
            _apply_chat_template(processor, m, add_generation_prompt=True)
            for m in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        return dict(
            processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        )

    def predict_from_inputs(self, inputs: dict) -> list[Prediction]:
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("MiMoVLAdapter.load() must run before predict()")

        processor = self._processor
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
            Prediction(bbox=_parse_mimo_bbox(text), score=None)
            for text in text_outputs
        ]

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        return self.predict_from_inputs(self.prepare_inputs(samples))

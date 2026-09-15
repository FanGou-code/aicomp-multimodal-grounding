"""InternVL3.5-8B trainable grounding adapter.

Verification status
-------------------
Prompt and modal placeholder contracts are pinned by unit tests. The GPU path
follows the official ``OpenGVLab/InternVL3_5-8B-HF``
transformers-native usage with [x1, y1, x2, y2] on a 0-1000 grid, but the
real GPU val slice should still be smoke-tested before a recorded run.
"""

from __future__ import annotations

import re
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
from aicomp_grounding.training_state import validated_prompt_length

MODEL_NAME = "OpenGVLab/InternVL3_5-8B-HF"
# ModelScope snapshot; the SOP download command pins this same revision.
MODEL_REVISION = "1c352b29d4066a61b465b5c6d044a1ebec1349ef"

# Official grounding evaluation allots 100 tokens; InternVL may prefix prose
# before the box, so keep the same budget here.
MAX_NEW_TOKENS = 100

INTERNVL_IMAGE_TOKEN = "<IMG_CONTEXT>"
# The HF processor owns the dynamic tile budget through
# preprocessor_config.json `max_patches`; InternVL3.5 no longer uses the old
# InternVL2 parameter name `max_num_tiles`.
MAX_PATCHES = 12

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
    """Full user question with numbered InternVL image-context slots."""
    return (
        f"Image-1: {INTERNVL_IMAGE_TOKEN}\n"
        f"Image-2: {INTERNVL_IMAGE_TOKEN}\n"
        f"Image-3: {INTERNVL_IMAGE_TOKEN}\n"
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
    supports_prepared_inputs = True

    def __init__(self):
        self.generation_config: dict[str, Any] = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "max_patches": MAX_PATCHES,
        }
        self._processor = None
        self._model = None

    def prompt_hash(self) -> str:
        import hashlib

        payload = {
            "modality_preamble": MODALITY_PREAMBLE,
            "grounding_prompt": GROUNDING_PROMPT_SUFFIX,
            "image_slots": [INTERNVL_IMAGE_TOKEN] * 3,
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
        source = require_local_model_path(model_path)

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor

        # InternVL3.5-HF controls the patch budget in preprocessor_config
        # (`max_patches`); do not pass the legacy InternVL2 max_num_tiles name.
        processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"

        try:
            from transformers import InternVLForConditionalGeneration

            model_class = InternVLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForImageTextToText

            model_class = AutoModelForImageTextToText

        try:
            model = model_class.from_pretrained(
                source,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = model_class.from_pretrained(
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
            "eval_batch_size": 1,
            "best_epoch_primary_metric": "acc_at_0_5",
            "python_version": RUNTIME_PYTHON_VERSION,
            "lora_targets": self.lora_target_modules(),
            "max_patches": MAX_PATCHES,
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
        text_config = getattr(self._model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            # Nested multimodal models keep the language model under
            # text_config; the outer config has no use_cache attribute, so
            # setting it there is a no-op and generation still builds a KV
            # cache during training forward passes.
            text_config.use_cache = False
        else:
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

        images = []
        for field in ("visible", "infrared", "depth"):
            with Image.open(data_root / item[field]) as opened:
                images.append(opened.convert("RGB"))
        query = item["query"]
        box = item["bbox"]
        answer = (
            "<box>["
            f"[{int(box[0] * 1000)},{int(box[1] * 1000)},"
            f"{int(box[2] * 1000)},{int(box[3] * 1000)}]"
            "]</box>"
        )
        prompt_messages = [
            {"role": "user", "content": build_grounding_question(query)}
        ]
        training_messages = [
            {"role": "user", "content": build_grounding_question(query)},
            {"role": "assistant", "content": answer},
        ]
        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        training_text = processor.apply_chat_template(
            training_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_inputs = processor(
            text=[prompt_text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = processor(
            text=[training_text],
            images=images,
            padding=True,
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

        # Current training config uses batch_size=1; returning the item directly
        # avoids processor.pad, which InternVLProcessor does not expose.
        if len(batch) == 1:
            item = dict(batch[0])
            item["input_ids"] = item["input_ids"].unsqueeze(0)
            item["attention_mask"] = item["attention_mask"].unsqueeze(0)
            item["labels"] = item["labels"].unsqueeze(0).long()
            return item

        pad_id = processor.tokenizer.pad_token_id
        max_length = max(item["input_ids"].size(0) for item in batch)
        if any("image_flags" not in item for item in batch):
            raise ValueError(
                "InternVL training batch is missing image_flags; "
                "the processor output does not match the expected contract"
            )
        input_ids = []
        labels = []
        attention_masks = []
        pixel_values = []
        image_flags = []
        for item in batch:
            padding = max_length - item["input_ids"].size(0)
            input_ids.append(
                torch.cat(
                    [
                        torch.full((padding,), pad_id, dtype=item["input_ids"].dtype),
                        item["input_ids"],
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        torch.full((padding,), -100, dtype=item["labels"].dtype),
                        item["labels"],
                    ]
                )
            )
            attention_masks.append(
                torch.cat(
                    [
                        torch.zeros(padding, dtype=item["attention_mask"].dtype),
                        item["attention_mask"],
                    ]
                )
            )
            pixel_values.append(item["pixel_values"])
            # image_flags is per-patch (flat), so concat — not stack — across samples.
            image_flags.append(item["image_flags"])
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels).long(),
            "attention_mask": torch.stack(attention_masks),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_flags": torch.cat(image_flags, dim=0),
        }

    def build_grounding_batch(self, samples: list[ModelInput], *, processor) -> dict:
        conversations = [
            [{"role": "user", "content": build_grounding_question(sample.query)}]
            for sample in samples
        ]
        texts = [
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        images = [
            image
            for sample in samples
            for image in (sample.visible, sample.infrared, sample.depth)
        ]
        return processor(text=texts, images=images, padding=True, return_tensors="pt")

    def decode_grounding_outputs(self, processor, generated_ids, prompt_len: int) -> list[str]:
        return processor.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=False,
        )

    def parse_grounding_text(self, text: str) -> list[float] | None:
        return parse_internvl_box(text)

    def prepare_inputs(self, samples: list[ModelInput]) -> dict:
        if self._processor is None:
            raise RuntimeError("InternVL35Adapter.load() must run before predict()")

        processor = self._processor
        conversations = [
            [{"role": "user", "content": build_grounding_question(sample.query)}]
            for sample in samples
        ]
        texts = [
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        images = [
            image
            for sample in samples
            for image in (sample.visible, sample.infrared, sample.depth)
        ]
        return dict(
            processor(text=texts, images=images, padding=True, return_tensors="pt")
        )

    def predict_from_inputs(self, inputs: dict) -> list[Prediction]:
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("InternVL35Adapter.load() must run before predict()")

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
            generated_tokens,
            skip_special_tokens=False,
        )
        return [
            Prediction(bbox=parse_internvl_box(text), score=None)
            for text in text_outputs
        ]

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        return self.predict_from_inputs(self.prepare_inputs(samples))

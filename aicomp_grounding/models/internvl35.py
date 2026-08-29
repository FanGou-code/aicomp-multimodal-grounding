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
from aicomp_grounding.config import RUNTIME_PYTHON_VERSION
from aicomp_grounding.models.base import ModelInput, Prediction

MODEL_NAME = "OpenGVLab/InternVL3_5-8B-HF"
MODEL_REVISION = "741a7d03020411e666c6109218ab71e08151ef86"

# Official grounding evaluation allots 100 tokens; InternVL may prefix prose
# before the box, so keep the same budget here.
MAX_NEW_TOKENS = 100

INTERNVL_IMAGE_TOKEN = "<IMG_CONTEXT>"

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


def extract_generated_tokens(generated_ids, prompt_lengths) -> list:
    """Strip each left-padded prompt using its own actual length."""
    return [row[length:] for row, length in zip(generated_ids, prompt_lengths)]


class InternVL35Adapter:
    name = "internvl35"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    supports_lora = True
    supports_prepared_inputs = True

    def __init__(self, *, max_num_tiles: int = 12):
        self.max_num_tiles = max_num_tiles
        self.generation_config: dict[str, Any] = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "max_num_tiles": max_num_tiles,
        }
        self._processor = None
        self._model = None
        self._grounding_prompt_lengths = None

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
            "lora_alpha": 48,
            "lora_dropout": 0.05,
            "compute_dtype": "bfloat16",
            "autocast": True,
            "eval_batch_size": 4,
            "best_epoch_primary_metric": "acc_at_0_5",
            "python_version": RUNTIME_PYTHON_VERSION,
            "lora_targets": self.lora_target_modules(),
            "max_num_tiles": self.max_num_tiles,
        }

    def lora_target_modules(self) -> list[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def load_for_training(
        self,
        *,
        device: str = "cuda",
        lora_path: Path | None = None,
        model_path: str | None = None,
    ):
        self.load(device=device, lora_path=lora_path, model_path=model_path)
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
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        if (
            inputs["input_ids"].shape[1] <= prompt_length
            or not bool(
                (
                    inputs["input_ids"][:, :prompt_length]
                    == prompt_inputs["input_ids"]
                ).all()
                .item()
            )
        ):
            raise ValueError(f"Training prompt is not an exact prefix for {item.get('key')}")
        labels = inputs["input_ids"].clone()
        labels[0, :prompt_length] = -100
        result = {key: value.squeeze(0) for key, value in inputs.items()}
        result["labels"] = labels.squeeze(0)
        return result

    def collate_training_batch(self, batch: list[dict], *, processor) -> dict:
        padded = processor.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = padded["labels"].long()
        return padded

    def build_grounding_batch(self, samples: list[ModelInput], *, processor) -> dict:
        messages = [
            {"role": "user", "content": build_grounding_question(sample.query)}
            for sample in samples
        ]
        texts = [
            processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        images = [
            image
            for sample in samples
            for image in (sample.visible, sample.infrared, sample.depth)
        ]
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        self._grounding_prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        return inputs

    def decode_grounding_outputs(self, processor, generated_ids, prompt_len: int) -> list[str]:
        lengths = self._grounding_prompt_lengths
        if lengths is None or len(lengths) != len(generated_ids):
            lengths = [int(prompt_len)] * len(generated_ids)
        text_outputs = []
        for row in extract_generated_tokens(generated_ids, lengths):
            batch = row.unsqueeze(0) if hasattr(row, "unsqueeze") else [row]
            text_outputs.append(
                processor.batch_decode(batch, skip_special_tokens=False)[0]
            )
        return text_outputs

    def parse_grounding_text(self, text: str) -> list[float] | None:
        return parse_internvl_box(text)

    def prepare_inputs(self, samples: list[ModelInput]) -> dict:
        if self._processor is None:
            raise RuntimeError("InternVL35Adapter.load() must run before predict()")

        processor = self._processor
        messages = [
            {"role": "user", "content": build_grounding_question(sample.query)}
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

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        text_outputs = []
        for row in extract_generated_tokens(generated_ids, prompt_lengths):
            batch = row.unsqueeze(0) if hasattr(row, "unsqueeze") else [row]
            text_outputs.append(
                processor.batch_decode(batch, skip_special_tokens=False)[0]
            )
        return [
            Prediction(bbox=parse_internvl_box(text), score=None)
            for text in text_outputs
        ]

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        return self.predict_from_inputs(self.prepare_inputs(samples))

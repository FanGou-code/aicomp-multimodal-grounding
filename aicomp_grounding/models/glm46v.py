"""GLM-4.6V-Flash trainable grounding adapter.

GLM-4.6V-Flash is the dense 9B member of the GLM-4.6V family (model_type
``glm4v``). The Hugging Face chat template accepts an ``enable_thinking``
kwarg; the grounding protocol disables it so the model emits only the box
token sequence. Bounding boxes use the tokenizer's added box-delimiter
tokens (ids 151361 / 151362) with integer coordinates on a 0-1000 grid
normalized by image width and height. The processor output schema
(mm_token_type_ids, image_grid_thw, pixel_values) mirrors
:mod:`aicomp_grounding.models.qwen3vl`, so the training collate is reused.

Local processor dry-run verified (no weights): the three-image chat template
renders with ``enable_thinking=False`` and the supervised target is an exact
prefix of the generation prompt. The GPU val slice should still be
smoke-tested before a recorded run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aicomp_grounding.bbox import quantize_bbox_1000, validate_bbox
from aicomp_grounding.config import RUNTIME_PYTHON_VERSION
from aicomp_grounding.models.base import ModelInput, Prediction, require_local_model_path
from aicomp_grounding.training_state import validated_prompt_length

MODEL_NAME = "zai-org/GLM-4.6V-Flash"
# ModelScope hosts the GLM family under the ZhipuAI org (not zai-org); the
# version below identifies the pre-download source. Execution only reads
# the explicit local --model-path; it does not contact either hub.
MODEL_REVISION = "a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41"
MODELSCOPE_NAME = "ZhipuAI/GLM-4.6V-Flash"

MAX_NEW_TOKENS = 32

# Tokenizer-added box delimiters (verified from the added vocab): ids
# 151361 and 151362. Built from chr() so the source carries no raw
# pipe-delimited special tokens that could confuse editors or diffs.
GLM_BOX_OPEN = chr(0x3C) + "|begin_of_box|" + chr(0x3E)
GLM_BOX_CLOSE = chr(0x3C) + "|end_of_box|" + chr(0x3E)

GLM_GROUNDING_SYSTEM_PROMPT = (
    "You are a visual grounding assistant. The images are ordered as visible "
    "RGB, infrared, and depth of the same scene. Output only "
    + GLM_BOX_OPEN
    + "x1,y1,x2,y2"
    + GLM_BOX_CLOSE
    + ", using integer coordinates from 0 to 1000 normalized by image width "
    "and height."
)
GLM_GROUNDING_USER_PROMPT = (
    "Help me to locate {query} in the image and give me its bounding boxes."
)

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

_GLM_BOX_PATTERN = re.compile(
    re.escape(GLM_BOX_OPEN)
    + r"\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*"
    + re.escape(GLM_BOX_CLOSE)
)
_GLM_BARE_PATTERN = re.compile(
    r"(?<![\w.+-])([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)(?!\w|\.\d)"
)


def format_glm_bbox(box) -> str:
    """Format normalized XYXY as GLM integer 0-1000 box tokens."""
    x1, y1, x2, y2 = quantize_bbox_1000(box)
    return f"{GLM_BOX_OPEN}{x1},{y1},{x2},{y2}{GLM_BOX_CLOSE}"


def parse_glm_box(text: str) -> list[float] | None:
    """Parse the GLM box token sequence (0-1000) into XYXY."""
    if not isinstance(text, str) or not text.strip():
        return None
    tagged = _GLM_BOX_PATTERN.findall(text)
    if len(tagged) == 1:
        values = [float(v) for v in tagged[0]]
    elif not tagged:
        candidates = list(_GLM_BARE_PATTERN.finditer(text))
        if len(candidates) != 1:
            return None
        bare = candidates[0]
        # Preserve the existing bare-coordinate fallback, including surrounding
        # prose, without taking a numeric substring or four entries from a
        # longer coordinate list.
        prefix, suffix = text[:bare.start()].rstrip(), text[bare.end():].lstrip()
        if re.search(r"[\d.]\s*,$", prefix) or re.match(r",\s*[+-]?\d", suffix):
            return None
        values = [float(v) for v in bare.groups()]
    else:
        return None
    if not all(0.0 <= v <= 1000.0 for v in values):
        return None
    return validate_bbox([v / 1000.0 for v in values])


def _apply_chat_template(processor, messages, *, add_generation_prompt: bool) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **CHAT_TEMPLATE_KWARGS,
    )


def _resolve_model_source(model_path: str | None) -> str:
    """Resolve the required pre-downloaded model directory."""
    return require_local_model_path(model_path)


def _build_messages(visible, infrared, depth, query, *, assistant_text=None):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GLM_GROUNDING_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": visible},
                {"type": "image", "image": infrared},
                {"type": "image", "image": depth},
                {"type": "text", "text": GLM_GROUNDING_USER_PROMPT.format(query=query)},
            ],
        },
    ]
    if assistant_text is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        )
    return messages


class Glm46VAdapter:
    name = "glm46v"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    supports_lora = True
    supports_prepared_inputs = True

    def __init__(self):
        self.generation_config: dict[str, Any] = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
        }
        self._processor = None
        self._model = None

    def prompt_hash(self) -> str:
        import hashlib
        import json

        payload = {
            "system_prompt": GLM_GROUNDING_SYSTEM_PROMPT,
            "grounding_prompt": GLM_GROUNDING_USER_PROMPT,
            "box_open": GLM_BOX_OPEN,
            "box_close": GLM_BOX_CLOSE,
            "image_count": 3,
            "enable_thinking": False,
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
        source = _resolve_model_source(model_path)

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Glm4vForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"
        print(
            f"[glm46v] image processor: {type(processor.image_processor).__name__}",
            flush=True,
        )

        try:
            model = Glm4vForConditionalGeneration.from_pretrained(
                source,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = Glm4vForConditionalGeneration.from_pretrained(
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
        visible, infrared, depth = images
        assistant_text = format_glm_bbox(item["bbox"])
        prompt_messages = _build_messages(visible, infrared, depth, item["query"])
        training_messages = _build_messages(
            visible, infrared, depth, item["query"], assistant_text=assistant_text
        )
        training_text = _apply_chat_template(
            processor, training_messages, add_generation_prompt=False
        )
        prompt_text = _apply_chat_template(
            processor, prompt_messages, add_generation_prompt=True
        )
        inputs = processor(
            text=[training_text],
            images=images,
            return_tensors="pt",
        )
        if "mm_token_type_ids" not in inputs:
            raise ValueError(
                f"GLM-4.6V training batch is missing mm_token_type_ids for {item.get('key')}"
            )
        prompt_inputs = processor(
            text=[prompt_text],
            images=images,
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
        ids, labels, attention, pixels, grids, mm_token_types = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for item in batch:
            if "mm_token_type_ids" not in item:
                raise ValueError("GLM-4.6V training batch is missing mm_token_type_ids")
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
        messages_list = [
            _build_messages(sample.visible, sample.infrared, sample.depth, sample.query)
            for sample in samples
        ]
        texts = [
            _apply_chat_template(processor, m, add_generation_prompt=True)
            for m in messages_list
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
        return parse_glm_box(text)

    def prepare_inputs(self, samples: list[ModelInput]) -> dict:
        if self._processor is None:
            raise RuntimeError("Glm46VAdapter.load() must run before predict()")

        processor = self._processor
        messages_list = [
            _build_messages(sample.visible, sample.infrared, sample.depth, sample.query)
            for sample in samples
        ]
        texts = [
            _apply_chat_template(processor, m, add_generation_prompt=True)
            for m in messages_list
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
            raise RuntimeError("Glm46VAdapter.load() must run before predict()")

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
            Prediction(bbox=parse_glm_box(text), score=None)
            for text in text_outputs
        ]

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        return self.predict_from_inputs(self.prepare_inputs(samples))

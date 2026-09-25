"""GLM-4.6V-Flash trainable grounding adapter.

GLM-4.6V-Flash is the dense 9B member of the GLM-4.6V family (model_type
``glm4v``). The Hugging Face chat template accepts an ``enable_thinking``
kwarg; the grounding protocol disables it so the model emits only the box
token sequence. Bounding boxes use the tokenizer's added box-delimiter
tokens (ids 151361 / 151362) with integer coordinates on a 0-1000 grid
normalized by image width and height. The processor output schema
(mm_token_type_ids, image_grid_thw, pixel_values) mirrors
:mod:`aicomp_grounding.serving.models.qwen3vl`, so the training collate is reused.

Local processor dry-run verified (no weights): the single-image chat template
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
from aicomp_grounding.serving.messages import (
    GLM_GROUNDING_PROMPT_PROTOCOL,
    GLM46V_SYSTEM_PROMPT,
    GLM46V_USER_TEMPLATE,
    grounding_prompt_hash,
)
from aicomp_grounding.serving.models.base import (
    ModelInput,
    Prediction,
    content_parts,
    language_model_lora_targets,
    require_local_model_path,
    run_generation,
)
from aicomp_grounding.serving.engine.training_state import validated_prompt_length

MODEL_NAME = "zai-org/GLM-4.6V-Flash"
# Weight snapshot recorded at adoption; fetch this revision before execution
# (repo and revision are listed in the README weight table). Execution only
# reads the explicit local --model-path and never contacts a hub.
MODEL_REVISION = "a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41"

MAX_NEW_TOKENS = 32

# Pixel budgets, in units of 28x28 pixels per vision token (patch 14 x merge 2),
# stated per frame so that the value recorded in the run identity means the same
# thing here as it does for the other adapters.
#
# Glm46VImageProcessor takes its budget as a `size` dict -- `min_pixels` /
# `max_pixels` passed to `AutoProcessor.from_pretrained` are dropped without
# warning -- and its `smart_resize` is called with `num_frames =
# temporal_factor`, so what it compares against `longest_edge` is `2 * h * w`.
# `load()` therefore doubles both budgets at the processor boundary
# (GLM_PIXEL_UNIT_FACTOR).  Handing it the per-frame numbers unmodified would
# halve the effective budget and shrink every frame: at 1920x1080, 2691 vision
# tokens down to 1508.
#
# The checkpoint's own preprocessor config allows `longest_edge = 28*28*12288`,
# which in the doubled unit is a per-frame budget of 4,816,896 -- twice the
# value below.  Neither the training nor the inference path overrode it, so
# GLM-4.6V ran on that wider budget until this pin was made to take effect.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 3072 * 28 * 28

#: Glm46VImageProcessor compares `temporal_factor * h * w` against
#: `size.longest_edge`; a per-frame budget is doubled at the boundary.
GLM_PIXEL_UNIT_FACTOR = 2

# Tokenizer-added box delimiters (verified from the added vocab): ids
# 151361 and 151362. Built from chr() so the source carries no raw
# pipe-delimited special tokens that could confuse editors or diffs.
GLM_BOX_OPEN = chr(0x3C) + "|begin_of_box|" + chr(0x3E)
GLM_BOX_CLOSE = chr(0x3C) + "|end_of_box|" + chr(0x3E)

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


def processor_pixel_kwargs(max_pixels: int) -> dict[str, Any]:
    """Convert a per-frame pixel budget into Glm46V's processor `size` dict.

    The processor counts `temporal_factor * h * w`, so the unit conversion lives
    here and is doubled once, at the boundary -- see the constants above.
    """
    return {
        "size": {
            "shortest_edge": GLM_PIXEL_UNIT_FACTOR * MIN_PIXELS,
            "longest_edge": GLM_PIXEL_UNIT_FACTOR * max_pixels,
        }
    }


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


def _apply_chat_template(
    processor,
    messages,
    *,
    add_generation_prompt: bool,
    template_kwargs: dict | None = None,
) -> str:
    kwargs = dict(CHAT_TEMPLATE_KWARGS)
    kwargs.update(template_kwargs or {})
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _resolve_model_source(model_path: str | None) -> str:
    """Resolve the required pre-downloaded model directory."""
    return require_local_model_path(model_path)


def _build_messages(visible, query, *, assistant_text=None):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GLM46V_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": visible},
                {"type": "text", "text": GLM46V_USER_TEMPLATE.format(query=query)},
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

    def __init__(self, *, max_pixels: int = MAX_PIXELS):
        # Per frame, i.e. in the same unit the other five adapters record;
        # `load()` converts it to the processor's doubled unit.
        self.max_pixels = max_pixels
        self.generation_config: dict[str, Any] = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "max_pixels": max_pixels,
        }
        self._processor = None
        self._model = None

    def prompt_hash(self) -> str:
        return grounding_prompt_hash(
            GLM_GROUNDING_PROMPT_PROTOCOL, GLM46V_SYSTEM_PROMPT, GLM46V_USER_TEMPLATE
        )

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
            **processor_pixel_kwargs(self.max_pixels),
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
            "gradient_checkpointing": True,
            "eval_batch_size": 1,
            "best_epoch_primary_metric": "acc_at_0_5",
            "python_version": RUNTIME_PYTHON_VERSION,
            "lora_targets": self.lora_target_modules(),
            "min_pixels": MIN_PIXELS,
            "max_pixels": self.max_pixels,
        }

    def lora_target_modules(self) -> str:
        # GLM-4.6V fuses the MLP gate and up projections into `gate_up_proj`, so
        # this adapter declares the projections its own language model exposes
        # instead of the split `gate_proj` / `up_proj` pair the other five use.
        # Widening it to include `gate_up_proj` is a recipe change, not a fix.
        return language_model_lora_targets(
            "q_proj", "k_proj", "v_proj", "o_proj", "down_proj"
        )

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

        with Image.open(data_root / item["visible"]) as opened:
            visible = opened.convert("RGB")
        images = [visible]
        assistant_text = format_glm_bbox(item["bbox"])
        prompt_messages = _build_messages(visible, item["query"])
        training_messages = _build_messages(
            visible, item["query"], assistant_text=assistant_text
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
            _build_messages(sample.visible, sample.query)
            for sample in samples
        ]
        texts = [
            _apply_chat_template(processor, m, add_generation_prompt=True)
            for m in messages_list
        ]
        images = [sample.visible for sample in samples]
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
            _build_messages(sample.visible, sample.query)
            for sample in samples
        ]
        texts = [
            _apply_chat_template(processor, m, add_generation_prompt=True)
            for m in messages_list
        ]
        images = [sample.visible for sample in samples]
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
            Prediction(bbox=parse_glm_box(text))
            for text in text_outputs
        ]

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        return self.predict_from_inputs(self.prepare_inputs(samples))

    def generate_messages(
        self,
        messages_list: list[list[dict]],
        *,
        max_new_tokens: int,
        temperature: float,
        skip_special_tokens: bool = False,
        sampling_kwargs: dict | None = None,
        template_kwargs: dict | None = None,
    ) -> list[str]:
        """Template and generate caller-built chat messages (ordinal module)."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Glm46VAdapter.load() must run before generate_messages()")

        processor = self._processor
        texts = [
            _apply_chat_template(
                processor, m, add_generation_prompt=True, template_kwargs=template_kwargs
            )
            for m in messages_list
        ]
        images = [
            part["image"]
            for messages in messages_list
            for part in content_parts(messages[-1])
            if part.get("type") == "image"
        ]
        kwargs: dict[str, object] = {"text": texts, "padding": True, "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        return run_generation(
            self._model,
            processor,
            dict(processor(**kwargs)),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            skip_special_tokens=skip_special_tokens,
            sampling_kwargs=sampling_kwargs,
        )

"""Qwen3-VL-32B-Instruct grounding adapter: second-generation dense sibling.

Same ``Qwen3VLForConditionalGeneration`` class, processor, grounding prompt
protocol, 0-1000 box parsing, and LoRA target modules as the 8B reference
adapter (``qwen3vl``). The model identity differs, so this subclasses the
reference and overrides ``name`` / ``model_name`` / ``model_revision``; the
load / predict paths are inherited unchanged. Training uses the same effective
batch size as the 8B reference, but keeps batch/eval memory conservative for
the larger dense model.

This keeps the historical 8B fingerprints byte-identical while giving the 32B
run a distinct identity.
"""

from __future__ import annotations

from typing import Any

from aicomp_grounding.models.qwen3vl import Qwen3VLAdapter

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
MODEL_REVISION = "0cfaf48183f594c314753d30a4c4974bc75f3ccb"


class Qwen3VL32Adapter(Qwen3VLAdapter):
    name = "qwen3vl32"
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION

    def training_hyperparameters(self) -> dict[str, Any]:
        params = super().training_hyperparameters()
        params["batch_size"] = 1
        params["gradient_accumulation_steps"] = 16
        params["eval_batch_size"] = 1
        return params

"""Model adapter registry: one name -> one grounding adapter."""

from __future__ import annotations

from typing import Any

from aicomp_grounding.models.base import GroundingAdapter, ModelInput, Prediction
from aicomp_grounding.models.groundingdino import GroundingDINOAdapter
from aicomp_grounding.models.internvl35 import InternVL35Adapter
from aicomp_grounding.models.mock import MockAdapter
from aicomp_grounding.models.qwen3vl import Qwen3VLAdapter

ADAPTERS: dict[str, type] = {
    "qwen3vl": Qwen3VLAdapter,
    "internvl35": InternVL35Adapter,
    "groundingdino": GroundingDINOAdapter,
    "mock": MockAdapter,
}


def get_adapter(name: str, **kwargs: Any) -> GroundingAdapter:
    """Instantiate the adapter registered under ``name``."""
    try:
        adapter_cls = ADAPTERS[name]
    except KeyError:
        valid = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown model {name!r}; available: {valid}") from None
    return adapter_cls(**kwargs)


def available_models() -> list[str]:
    return sorted(ADAPTERS)


__all__ = [
    "ADAPTERS",
    "GroundingAdapter",
    "ModelInput",
    "Prediction",
    "available_models",
    "get_adapter",
]

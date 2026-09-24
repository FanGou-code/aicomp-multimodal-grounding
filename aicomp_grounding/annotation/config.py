"""Annotation-side configuration: teacher model identity and index location."""

from __future__ import annotations

from pathlib import Path

ANNOTATION_PROVIDER = "zhipu"
ANNOTATION_MODEL_NAME = "glm-4.6v"
ANNOTATION_MODEL_REVISION = "2025-12-08"
ANNOTATION_MODEL_WEIGHTS_URL = "https://huggingface.co/zai-org/GLM-4.6V"
ANNOTATION_MODEL_LICENSE = "MIT"
ANNOTATION_API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def resolve_index_dir(data_root: Path, index_dir: Path | None = None) -> Path:
    """Separate source indexes from image roots; accept the legacy co-located layout."""
    if index_dir is not None:
        return Path(index_dir).resolve()
    legacy = Path(data_root) / "indexes"
    if legacy.is_dir():
        return legacy.resolve()
    return Path(__file__).resolve().parents[2] / "data" / "indexes"

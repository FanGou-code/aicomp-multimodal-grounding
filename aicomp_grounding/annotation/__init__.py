"""Shared utilities for the RGBDT visual-grounding project."""

from .bbox import (
    compute_iou,
    format_qwen_bbox,
    normalize_pixel_bbox,
    parse_bbox_from_text,
    validate_bbox,
)

__all__ = [
    "compute_iou",
    "format_qwen_bbox",
    "normalize_pixel_bbox",
    "parse_bbox_from_text",
    "validate_bbox",
]

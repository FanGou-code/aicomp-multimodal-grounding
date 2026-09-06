"""Source fingerprint for immutable annotation inputs (deliberately ignores Query)."""

from __future__ import annotations

from foundry.artifacts import stable_json_hash


def source_fingerprint(dataset: dict) -> str:
    """Fingerprint the annotation-source identity of a dataset index.

    Only the immutable modality references and geometry participate; the
    ``query`` field is excluded so a text-only rewrite does not change the
    identity of the images being annotated.
    """
    identity = {}
    for key in sorted(dataset):
        item = dataset[key]
        identity[key] = {
            field: item.get(field)
            for field in ("visible", "infrared", "depth", "bbox", "width", "height")
        }
    return stable_json_hash(identity)

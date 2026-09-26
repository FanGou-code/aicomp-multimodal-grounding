"""Canonical official-Test contract validation."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from aicomp_grounding.artifacts import stable_json_hash

OFFICIAL_TEMPLATE_FIELDS = frozenset({"visible", "infrared", "depth", "query"})
OFFICIAL_MODALITY_DIRS = {
    "visible": "visible",
    "infrared": "infrared",
    "depth": "depth",
}
OFFICIAL_QUERY_COUNT = 5690
OFFICIAL_TEMPLATE_CANONICAL_SHA256 = (
    "9d0f805728859fb51f47be4d2892f7c8f12f16ab17a9b5a79e49a433b4247ed2"
)
def validate_official_test_template(
    template: dict,
    *,
    expected_query_count: int | None = OFFICIAL_QUERY_COUNT,
    expected_template_sha256: str | None = OFFICIAL_TEMPLATE_CANONICAL_SHA256,
) -> None:
    """Reject processed indexes and modified official Test templates."""
    if not isinstance(template, dict) or not template:
        raise ValueError("Official test template is empty or is not a JSON object")
    if expected_query_count is not None and len(template) != expected_query_count:
        raise ValueError(
            f"Official test template must contain {expected_query_count} queries, "
            f"got {len(template)}"
        )
    for query_id, item in template.items():
        if not isinstance(query_id, str) or not re.fullmatch(r"\d{6}_\d{3}", query_id):
            raise ValueError(f"Invalid official Query ID: {query_id!r}")
        if not isinstance(item, dict) or set(item) != OFFICIAL_TEMPLATE_FIELDS:
            raise ValueError(
                f"Official template item {query_id!r} must contain exactly "
                f"{sorted(OFFICIAL_TEMPLATE_FIELDS)} and no bbox"
            )
        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValueError(f"Official template item {query_id!r} has an empty query")
        stems = []
        for field, directory in OFFICIAL_MODALITY_DIRS.items():
            value = item[field]
            if not isinstance(value, str) or "\\" in value:
                raise ValueError(f"Official template item {query_id!r} has invalid {field}")
            path = PurePosixPath(value)
            if len(path.parts) != 3 or path.parts[:2] != ("Images", directory):
                raise ValueError(
                    f"{query_id!r} is not from the untouched official template: "
                    f"invalid {field} path {value!r}"
                )
            stems.append(path.stem)
        if len(set(stems)) != 1:
            raise ValueError(f"Official modalities refer to different scenes for {query_id!r}")
    if (
        expected_template_sha256 is not None
        and stable_json_hash(template) != expected_template_sha256
    ):
        raise ValueError(
            "Official test template content does not match the pinned canonical SHA-256"
        )


"""Canonical official-Test and processed-index contract validation."""

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
OFFICIAL_QUERY_COUNT = 9555
OFFICIAL_TEMPLATE_CANONICAL_SHA256 = (
    "8fae701890bbbf05099e11ac8b2a3ead18990a496449355c9f09825d88ccfbbf"
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


def validate_processed_test_index(
    processed: dict,
    official: dict,
    *,
    expected_query_count: int | None = OFFICIAL_QUERY_COUNT,
    expected_template_sha256: str | None = OFFICIAL_TEMPLATE_CANONICAL_SHA256,
) -> None:
    """Require the inference index to be an exact path mapping of the official template."""
    validate_official_test_template(
        official,
        expected_query_count=expected_query_count,
        expected_template_sha256=expected_template_sha256,
    )
    if not isinstance(processed, dict) or not processed:
        raise ValueError("Processed test index is empty or is not a JSON object")
    if set(processed) != set(official):
        missing = sorted(set(official) - set(processed))
        extra = sorted(set(processed) - set(official))
        raise ValueError(
            "Processed test index IDs do not match the official template: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    for query_id, source in official.items():
        item = processed[query_id]
        if not isinstance(item, dict) or set(item) != OFFICIAL_TEMPLATE_FIELDS:
            raise ValueError(
                f"Processed test item {query_id!r} must contain exactly "
                f"{sorted(OFFICIAL_TEMPLATE_FIELDS)}"
            )
        expected = {
            "visible": f"Test/{source['visible']}",
            "infrared": f"Test/{source['infrared']}",
            "depth": (
                "Processed/Test/depth_jet/"
                f"{PurePosixPath(source['depth']).name}"
            ),
            "query": source["query"],
        }
        mismatches = [field for field in OFFICIAL_TEMPLATE_FIELDS if item[field] != expected[field]]
        if mismatches:
            details = ", ".join(
                f"{field}={item[field]!r} (expected {expected[field]!r})"
                for field in sorted(mismatches)
            )
            raise ValueError(
                f"Processed test item {query_id!r} does not map to the official template: "
                f"{details}"
            )

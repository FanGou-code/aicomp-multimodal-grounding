"""Tests for committed-volume and deep image fingerprint policies."""

from __future__ import annotations

import copy
import unittest

from aicomp_grounding.images import (
    is_trusted_image_fingerprint,
    trusted_dataset_image_fingerprint,
)


def _dataset() -> dict:
    return {
        "001_00000001": {
            "visible": "Raw/001/color/00000001.png",
            "infrared": "Raw/001/infrared/00000001.png",
            "depth": "Raw/001/depth/00000001.png",
            "width": 1920,
            "height": 1080,
        }
    }


class TrustedImageFingerprintTests(unittest.TestCase):
    def test_fingerprint_needs_no_image_files_and_is_deterministic(self):
        dataset = _dataset()
        first = trusted_dataset_image_fingerprint(
            dataset,
            dataset,
            require_recorded_size=True,
        )
        second = trusted_dataset_image_fingerprint(
            copy.deepcopy(dataset),
            list(reversed(dataset)),
            require_recorded_size=True,
        )
        self.assertEqual(first, second)
        self.assertTrue(is_trusted_image_fingerprint(first))

    def test_path_or_recorded_size_change_changes_fingerprint(self):
        dataset = _dataset()
        original = trusted_dataset_image_fingerprint(
            dataset,
            dataset,
            require_recorded_size=True,
        )
        changed_path = copy.deepcopy(dataset)
        changed_path["001_00000001"]["depth"] = (
            "Raw/001/depth/changed.png"
        )
        changed_size = copy.deepcopy(dataset)
        changed_size["001_00000001"]["width"] = 1280
        for changed in (changed_path, changed_size):
            self.assertNotEqual(
                original,
                trusted_dataset_image_fingerprint(
                    changed,
                    changed,
                    require_recorded_size=True,
                ),
            )

    def test_noncanonical_or_duplicate_modality_paths_are_rejected(self):
        noncanonical = _dataset()
        noncanonical["001_00000001"]["depth"] = "../depth.png"
        with self.assertRaisesRegex(ValueError, "non-canonical"):
            trusted_dataset_image_fingerprint(
                noncanonical,
                noncanonical,
                require_recorded_size=True,
            )

        duplicate = _dataset()
        duplicate["001_00000001"]["depth"] = duplicate["001_00000001"]["visible"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            trusted_dataset_image_fingerprint(
                duplicate,
                duplicate,
                require_recorded_size=True,
            )


if __name__ == "__main__":
    unittest.main()

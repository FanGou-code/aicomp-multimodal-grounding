"""Cross-repo consistency tests against shared_vectors.json.

These tests assert that the local implementation of each shared function
produces exactly the same output as the frozen test vectors. If either repo
changes its algorithm, this test will fail and the vectors must be regenerated
deliberately (not silently).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from aicomp_grounding.artifacts import key_hash, stable_json_hash
from aicomp_grounding.bbox import format_qwen_bbox, quantize_bbox_1000
from aicomp_grounding.images import is_trusted_image_fingerprint, trusted_dataset_image_fingerprint
from aicomp_grounding.sequence import source_fingerprint

VECTORS = json.loads((Path(__file__).parent / "shared_vectors.json").read_text())


class SharedVectorTests(unittest.TestCase):
    def test_stable_json_hash(self):
        self.assertEqual(stable_json_hash(VECTORS["stable_json_hash"]["input"]),
                         VECTORS["stable_json_hash"]["expected"])

    def test_key_hash(self):
        self.assertEqual(key_hash(VECTORS["key_hash"]["input"]),
                         VECTORS["key_hash"]["expected"])

    def test_source_fingerprint(self):
        self.assertEqual(source_fingerprint(VECTORS["stable_json_hash"]["input"]),
                         VECTORS["source_fingerprint"]["expected"])

    def test_trusted_dataset_image_fingerprint(self):
        data = VECTORS["stable_json_hash"]["input"]
        self.assertEqual(
            trusted_dataset_image_fingerprint(data, sorted(data), require_recorded_size=True),
            VECTORS["trusted_dataset_image_fingerprint"]["expected"],
        )

    def test_is_trusted_image_fingerprint(self):
        for case in VECTORS["is_trusted_image_fingerprint"]["cases"]:
            with self.subTest(input=case["input"]):
                self.assertEqual(is_trusted_image_fingerprint(case["input"]), case["expected"])

    def test_quantize_bbox_1000(self):
        for case in VECTORS["quantize_bbox_1000"]["cases"]:
            with self.subTest(input=case["input"]):
                self.assertEqual(quantize_bbox_1000(case["input"]), case["expected"])

    def test_format_qwen_bbox(self):
        for case in VECTORS["format_qwen_bbox"]["cases"]:
            with self.subTest(input=case["input"]):
                self.assertEqual(
                    format_qwen_bbox(case["input"], special_tokens=case["special_tokens"]),
                    case["expected"],
                )


if __name__ == "__main__":
    unittest.main()

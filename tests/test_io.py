import json
import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.io import atomic_write_json, load_json, require_distinct_paths


class JsonIoTests(unittest.TestCase):
    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "result.json"
            atomic_write_json(path, {"sample": {"bbox": [0.1, 0.2, 0.3, 0.4]}})
            self.assertEqual(load_json(path)["sample"]["bbox"], [0.1, 0.2, 0.3, 0.4])
            json.loads(path.read_text(encoding="utf-8"))

    def test_refuses_in_place_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            with self.assertRaises(ValueError):
                require_distinct_paths(path, path)

    def test_duplicate_json_keys_are_rejected_at_any_level(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "duplicate.json"
            for payload in (
                '{"sample": 1, "sample": 2}',
                '{"sample": {"query": "first", "query": "second"}}',
            ):
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                        load_json(path)


if __name__ == "__main__":
    unittest.main()

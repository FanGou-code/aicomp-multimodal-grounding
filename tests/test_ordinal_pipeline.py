"""End-to-end check of the ordinal entry points on CPU, driven by the mock adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from aicomp_grounding.ordinal.enumerate import main as enumerate_main
from aicomp_grounding.ordinal.resolve import main as resolve_main

QUERY_ID = "000001_001"
FIRST = [0.10, 0.10, 0.20, 0.30]
SECOND = [0.40, 0.10, 0.50, 0.30]


def _write_fixture(root: Path) -> Path:
    images = root / "data" / "Test" / "Images"
    for name in ("visible", "infrared", "depth"):
        (images / name).mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(images / "visible" / "000001.jpg")
    Image.new("RGB", (4, 4), (200, 200, 200)).save(images / "infrared" / "000001.png")
    Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(images / "depth" / "000001.png")

    queries = root / "queries.json"
    queries.write_text(
        json.dumps(
            {
                QUERY_ID: {
                    "visible": "Images/visible/000001.jpg",
                    "infrared": "Images/infrared/000001.png",
                    "depth": "Images/depth/000001.png",
                    "query": "the second person from the left",
                }
            }
        ),
        encoding="utf-8",
    )
    return queries


class MockOrdinalPipelineTests(unittest.TestCase):
    def _run(self, main_func, argv: list[str]) -> None:
        with mock.patch.object(sys, "argv", argv):
            main_func()

    def test_the_chain_replaces_the_box_when_two_models_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = _write_fixture(root)
            enum_root = root / "enum"
            enum_dirs = []
            for tag in ("m1", "m2", "m3"):
                before = set(enum_root.glob("*")) if enum_root.is_dir() else set()
                self._run(enumerate_main, [
                    "enumerate", "--model", "mock",
                    "--test-json", str(queries), "--data-dir", str(root / "data"),
                    "--output-dir", str(enum_root), "--run-tag", tag,
                    "--temperature", "0.7", "--batch-save", "1",
                ])
                enum_dirs.append(next(iter(set(enum_root.glob("*")) - before)))

            self.assertEqual(len(set(enum_dirs)), 3)
            for directory in enum_dirs:
                parsed = json.loads((directory / "parse.json").read_text(encoding="utf-8"))
                self.assertEqual(parsed[QUERY_ID]["intent"]["selection"]["mode"], "rank")
                instances = json.loads((directory / "instances.json").read_text(encoding="utf-8"))
                self.assertEqual(len(instances[QUERY_ID]["runs"]), 2)

            prediction_files = []
            for index in range(3):
                path = root / f"pred_{index}.json"
                path.write_text(json.dumps({QUERY_ID: list(SECOND)}), encoding="utf-8")
                prediction_files.append(path)

            argv = [
                "resolve", "--test-json", str(queries), "--data-dir", str(root / "data"),
                "--output-dir", str(root / "ordinal"),
            ]
            for directory, prediction in zip(enum_dirs, prediction_files, strict=True):
                argv += ["--enum-run", str(directory), "--predictions", str(prediction)]
            self._run(resolve_main, argv)

            out_dir = next(iter((root / "ordinal").glob("*")))
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            stats = metadata["stats"]
            self.assertEqual(stats["rank_queries"], 1)
            self.assertEqual(stats["adopted"], 1)
            self.assertEqual(stats["replaced_per_model"], [1, 1, 1])
            self.assertEqual(len(stats["outputs"]), 3)
            for entry in stats["outputs"]:
                written = json.loads((out_dir / entry["file"]).read_text(encoding="utf-8"))
                self.assertEqual(written[QUERY_ID], FIRST)

    def test_a_single_model_alone_leaves_the_box_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = _write_fixture(root)
            self._run(enumerate_main, [
                "enumerate", "--model", "mock",
                "--test-json", str(queries), "--data-dir", str(root / "data"),
                "--output-dir", str(root / "enum"), "--temperature", "0.7",
            ])
            enum_dir = next(iter((root / "enum").glob("*")))
            prediction = root / "pred.json"
            prediction.write_text(json.dumps({QUERY_ID: list(SECOND)}), encoding="utf-8")

            self._run(resolve_main, [
                "resolve", "--enum-run", str(enum_dir), "--predictions", str(prediction),
                "--test-json", str(queries), "--data-dir", str(root / "data"),
                "--output-dir", str(root / "ordinal"),
            ])
            out_dir = next(iter((root / "ordinal").glob("*")))
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["stats"]["adopted"], 0)
            self.assertEqual(metadata["stats"]["replaced_not_adopted"], 1)
            written = json.loads((out_dir / metadata["stats"]["outputs"][0]["file"]).read_text(encoding="utf-8"))
            self.assertEqual(written[QUERY_ID], SECOND)

    def test_enumerate_refuses_a_zero_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = _write_fixture(root)
            with self.assertRaises(SystemExit):
                self._run(enumerate_main, [
                    "enumerate", "--model", "mock",
                    "--test-json", str(queries), "--data-dir", str(root / "data"),
                    "--temperature", "0",
                ])


if __name__ == "__main__":
    unittest.main()

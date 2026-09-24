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

from aicomp_grounding.serving.ordinal import run as ordinal_run
from aicomp_grounding.serving.ordinal.enumerate import main as enumerate_main
from aicomp_grounding.serving.ordinal.resolve import main as resolve_main

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

    def _enumerate(self, root: Path, queries: Path, *extra: str) -> Path:
        self._run(enumerate_main, [
            "enumerate", "--model", "mock",
            "--test-json", str(queries), "--data-dir", str(root / "data"),
            "--output-dir", str(root / "enum"), "--batch-save", "1",
            *extra,
        ])
        return next(iter((root / "enum").glob("*")))

    def test_the_chain_replaces_the_box_and_keeps_the_thinking_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = _write_fixture(root)
            enum_dir = self._enumerate(root, queries)

            parsed = json.loads((enum_dir / "parse.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed[QUERY_ID]["intent"]["selection"]["mode"], "rank")
            instances = json.loads((enum_dir / "instances.json").read_text(encoding="utf-8"))
            self.assertEqual(instances[QUERY_ID]["payload"]["count"], 2)
            self.assertEqual(ordinal_run.load_thinking(enum_dir).get(QUERY_ID, "").count("2"), 1)

            prediction = root / "pred.json"
            prediction.write_text(json.dumps({QUERY_ID: list(SECOND)}), encoding="utf-8")

            self._run(resolve_main, [
                "resolve", "--enum-run", str(enum_dir), "--predictions", str(prediction),
                "--test-json", str(queries), "--data-dir", str(root / "data"),
                "--output-dir", str(root / "ordinal"),
            ])

            out_dir = next(iter((root / "ordinal").glob("*")))
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["stats"]["rank_queries"], 1)
            self.assertEqual(metadata["stats"]["adopted"], 1)
            self.assertEqual(len(metadata["stats"]["outputs"]), 1)
            entry = metadata["stats"]["outputs"][0]
            written = json.loads((out_dir / entry["file"]).read_text(encoding="utf-8"))
            self.assertEqual(written[QUERY_ID], FIRST)

    def test_enumerating_without_thinking_still_replaces_the_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = _write_fixture(root)
            enum_dir = self._enumerate(root, queries, "--no-think")
            self.assertEqual(ordinal_run.load_thinking(enum_dir), {})

            prediction = root / "pred.json"
            prediction.write_text(json.dumps({QUERY_ID: list(SECOND)}), encoding="utf-8")
            self._run(resolve_main, [
                "resolve", "--enum-run", str(enum_dir), "--predictions", str(prediction),
                "--test-json", str(queries), "--data-dir", str(root / "data"),
                "--output-dir", str(root / "ordinal"),
            ])
            out_dir = next(iter((root / "ordinal").glob("*")))
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["stats"]["adopted"], 1)

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

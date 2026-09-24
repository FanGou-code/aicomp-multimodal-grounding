"""End-to-end resume tests: an interrupted API stage must not pay twice.

Both API stages journal every finished item as it arrives and reuse the journal
on the next run. These tests drive the real CLI entry points with the network
boundary stubbed, so what is exercised is the wiring the operator actually runs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def _fake_response(payload: dict):
    class _Response:
        content = json.dumps(payload)
        record = {"usage": {"total_tokens": 2}}

    return _Response()


class _CountingClient:
    """Stands in for the HTTP client; each call answers with a fresh sentence."""

    def __init__(self, counter: list) -> None:
        self.counter = counter

    def complete(self, **_kwargs):
        self.counter[0] += 1
        return _fake_response({"query": f"the car number {self.counter[0]}"})


class AssembleResumeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.calls = [0]

    def tearDown(self):
        self._tmp.cleanup()

    def _fixture(self, frames: int = 3) -> None:
        data_root = self.root / "dataset"
        (data_root / "Train/001/color").mkdir(parents=True)
        (self.root / "index").mkdir()
        index = {}
        for i in range(1, frames + 1):
            sample_id = f"001_{i:08d}"
            visible = f"Train/001/color/{i:08d}.png"
            Image.new("RGB", (64, 48)).save(data_root / visible)
            index[sample_id] = {"visible": visible, "bbox": [0.1, 0.5, 0.2, 0.7]}
        (self.root / "index/train.json").write_text(json.dumps(index), encoding="utf-8")

        results = {}
        for sample_id in index:
            results[sample_id.split("_")[0]] = {
                "status": "completed",
                "selected": sorted(index),
                "frames": {
                    sample_id: {
                        "status": "completed",
                        "findall": {
                            "status": "completed", "mode": "instances", "attempts": 1,
                            "objects": [
                                {"i": 1, "category": "car", "bbox": [0.1, 0.5, 0.2, 0.7]},
                                {"i": 2, "category": "car", "bbox": [0.4, 0.5, 0.5, 0.7]},
                            ],
                        },
                    }
                },
            }
        census = self.root / "census"
        census.mkdir()
        (census / "merged.json").write_text(
            json.dumps({"metadata": {"run_id": "census_test"}, "results": results}),
            encoding="utf-8",
        )

    def _run(self, *extra: str) -> None:
        from scripts import assemble_queries

        argv = [
            "assemble_queries",
            "--census-run", str(self.root / "census"),
            "--data-root", str(self.root / "dataset"),
            "--index-dir", str(self.root / "index"),
            "--output-root", str(self.root / "out"),
            "--run-tag", "asm-test",
            "--split", "train",
            "--api-key", "fake",
            "--show", "0",
            *extra,
        ]

        with patch("sys.argv", argv), \
             patch.object(assemble_queries, "client_for", lambda *_a, **_k: _CountingClient(self.calls)):
            assemble_queries.main()

    @property
    def _manifest(self) -> dict:
        return json.loads((self.root / "out/asm-test/assembly.json").read_text(encoding="utf-8"))

    def test_the_journal_is_written_as_the_run_goes(self):
        self._fixture()
        self._run()
        lines = (self.root / "out/asm-test/sentences.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), len(self._manifest["records"]))
        for line in lines:
            row = json.loads(line)
            self.assertIn("#", row["item_id"])
            self.assertTrue(row["query"])

    def test_a_second_run_reuses_the_journal_and_calls_nothing(self):
        self._fixture()
        self._run()
        first = self._manifest["records"]
        calls_after_first = self.calls[0]

        self._run()
        self.assertEqual(self.calls[0], calls_after_first, "resume must not re-ask the teacher")
        self.assertEqual(self._manifest["records"], first)

    def test_force_starts_over(self):
        self._fixture()
        self._run()
        calls_after_first = self.calls[0]

        self._run("--force")
        self.assertGreater(self.calls[0], calls_after_first)

    def test_an_existing_dir_without_resume_or_force_is_refused(self):
        self._fixture()
        self._run()
        with self.assertRaises(SystemExit):
            self._run("--no-resume")


class ReverseResumeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "dataset").mkdir()
        Image.new("RGB", (64, 48)).save(self.root / "dataset/frame.png")
        self.queries_path = self.root / "queries.json"
        self.queries_path.write_text(
            json.dumps({f"q{i}": {"query": f"the {i}th car", "visible": "frame.png"} for i in range(1, 6)}),
            encoding="utf-8",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, calls: list, **options) -> dict:
        from scripts import run_reverse

        def resolve_one(_clients, _data_root, query, _visible):
            calls.append(query)
            return {"bbox": [0.1, 0.2, 0.3, 0.4], "route": "kernel", "calls": [], "thinking": []}

        with patch.object(run_reverse, "resolve_one", resolve_one), \
             patch.object(run_reverse, "client_for", lambda *_a, **_k: object()):
            return run_reverse.run_reverse(
                queries_path=self.queries_path,
                data_root=self.root / "dataset",
                output_root=self.root / "out",
                run_tag="rev-test",
                api_key="fake",
                **options,
            )

    def test_the_journal_is_written_and_reused(self):
        calls: list = []
        self._run(calls)
        self.assertEqual(len(calls), 5)
        journal = self._run_dir() / "results.jsonl"
        self.assertEqual(len(journal.read_text(encoding="utf-8").splitlines()), 5)

        resumed: list = []
        self._run(resumed)
        self.assertEqual(resumed, [], "resume must not re-ask the teacher")
        predictions = json.loads((self._run_dir() / "predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(predictions), 5)

    def test_limit_only_asks_for_the_first_n_and_resume_finishes_the_rest(self):
        calls: list = []
        self._run(calls, limit=2)
        self.assertEqual(len(calls), 2)

        rest: list = []
        self._run(rest)
        self.assertEqual(len(rest), 3)
        self.assertTrue((self._run_dir() / "results.jsonl").is_file())

    def test_force_asks_everything_again(self):
        calls: list = []
        self._run(calls)
        forced: list = []
        self._run(forced, force=True)
        self.assertEqual(len(forced), 5)

    def _run_dir(self) -> Path:
        from scripts.run_reverse import build_reverse_plan

        plan = build_reverse_plan(queries_path=self.queries_path, run_tag="rev-test")
        return self.root / "out" / plan["metadata"]["run_id"]


if __name__ == "__main__":
    unittest.main()

"""Tests for the census review server (foundry.review)."""

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# The admin shell exports http_proxy without a 127.* no_proxy exemption;
# route localhost test requests directly so the real review server (8788)
# and the proxy are both out of the loop.
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from foundry.review.census_session import build_assembly_session
from foundry.review.server import create_server
from foundry.review.store import AnnotationStore


def make_census_run(tmp: Path) -> tuple[Path, Path]:
    """Minimal census run + dataset index with one frame and two objects."""
    run_dir = tmp / "census_test0000"
    run_dir.mkdir(parents=True)
    objects = [
        {"i": 1, "category": "swan", "bbox": [0.10, 0.40, 0.20, 0.60]},
        {"i": 2, "category": "swan", "bbox": [0.50, 0.40, 0.60, 0.60]},
    ]
    frame = {
        "findall_1": {"status": "completed", "attempts": 1, "error": "", "objects": objects},
        "findall_2": {"status": "completed", "attempts": 1, "error": "", "objects": objects},
        "status": "completed",
        "error": "",
        "agreement": {"matched": 2, "count_a": 2, "count_b": 2, "count_agree": True, "jaccard": 1.0},
        "candidate": {"sample_id": "070_00000001", "frame_no": 1, "count": 2},
    }
    merged = {
        "metadata": {"run_id": "census_test0000", "split": "train"},
        "results": {
            "070": {
                "status": "completed",
                "frames": {"070_00000001": frame},
                "selected": ["070_00000001"],
            }
        },
    }
    (run_dir / "merged.json").write_text(json.dumps(merged), encoding="utf-8")

    data_root = tmp / "data"
    index_dir = data_root / "indexes"
    index_dir.mkdir(parents=True)
    (index_dir / "train.json").write_text(
        json.dumps({"070_00000001": {"visible": "Train/070/color/00000001.png",
                                     "bbox": [0.10, 0.40, 0.20, 0.60]}}),
        encoding="utf-8",
    )
    return run_dir, data_root


class ReviewServerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.run_dir, self.data_root = make_census_run(self.tmp)
        self.review_root = self.tmp / "review"
        self.server, self.state = create_server(
            census_run_dir=self.run_dir,
            data_root=self.data_root,
            review_root=self.review_root,
            host="127.0.0.1",
            port=0,
        )
        # The accept loop must actually run, or HTTP requests queue in the
        # kernel backlog forever (the earlier suite-wide hang).
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=5)
        self._tmp.cleanup()

    def test_session_shape_and_seeding(self):
        payload = self.state.session_payload()
        self.assertEqual(payload["mode"], "census-review")
        self.assertEqual(payload["total_items"], 2)
        self.assertEqual(payload["total_frames"], 1)
        self.assertEqual(payload["human_annotated"], 0)
        item = payload["items"][0]
        self.assertEqual(item["id"], "070_00000001#01")
        self.assertEqual(item["ordinal"], 1)
        self.assertEqual(item["frame_id"], "070_00000001")
        self.assertEqual(item["gt_bbox"], [0.10, 0.40, 0.20, 0.60])
        self.assertEqual(item["annotator"], "glm-4.6v")  # teacher box seeded
        self.assertAlmostEqual(item["bbox"][0], 0.10)

    def test_seeding_idempotent_across_restart(self):
        first = self.state.session_payload()
        self.server.server_close()
        server2, state2 = create_server(
            census_run_dir=self.run_dir,
            data_root=self.data_root,
            review_root=self.review_root,
            host="127.0.0.1",
            port=0,
        )
        threading.Thread(target=server2.serve_forever, daemon=True).start()
        try:
            self.assertEqual(state2.session["stats"]["seeded"], 0)
            self.assertEqual(state2.session["stats"]["already_seeded"], 2)
            # Teacher boxes survive the restart untouched.
            self.assertEqual(state2.session_payload()["items"][0]["bbox"], first["items"][0]["bbox"])
        finally:
            server2.shutdown()
            server2.server_close()

    def test_human_adjustment_persists_and_counts_as_reviewed(self):
        item_id = "070_00000001#01"
        request = urllib.request.Request(
            f"{self.base_url}/api/item/{urllib.parse.quote(item_id)}/bbox",
            data=json.dumps({"bbox": [0.11, 0.41, 0.21, 0.61], "annotator": "fang0"}).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(request) as resp:
            body = json.loads(resp.read())
            self.assertEqual(body["bbox"], [0.11, 0.41, 0.21, 0.61])
        payload = self.state.session_payload()
        first = next(e for e in payload["items"] if e["id"] == item_id)
        self.assertEqual(first["annotator"], "fang0")
        self.assertEqual(payload["human_annotated"], 1)
        self.assertEqual(payload["reviewed_frames"], 0)  # object #2 still AI-pending

    def test_put_requires_annotator(self):
        request = urllib.request.Request(
            f"{self.base_url}/api/item/070_00000001%2301/bbox",
            data=json.dumps({"bbox": [0.1, 0.4, 0.2, 0.6]}).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        self.assertEqual(ctx.exception.code, 400)



class ReviewReportTest(unittest.TestCase):
    def test_report_counts_adjustments_and_iou(self):
        import json as _json
        import tempfile as _tempfile
        from pathlib import Path as _Path

        from scripts.review_report import build_report

        with _tempfile.TemporaryDirectory() as tmp:
            tmp = _Path(tmp)
            run_dir, data_root = make_census_run(tmp)
            review_root = tmp / "review"
            server, state = create_server(
                census_run_dir=run_dir, data_root=data_root,
                review_root=review_root, host="127.0.0.1", port=0,
            )
            store = state.session["store"]
            # Human adjusts #01 strongly, confirms #02 untouched.
            store.set("070_00000001#01", [0.30, 0.40, 0.40, 0.60], annotator="fang0")
            server.server_close()

            report = build_report(run_dir, review_root)
            self.assertEqual(report["totals"]["total_items"], 2)
            self.assertEqual(report["totals"]["human_adjusted"], 1)
            self.assertEqual(report["totals"]["sequences_affected"], 1)
            seq = report["sequences"]["070"]
            adjusted = [r for r in seq["items"] if r["human_adjusted"]]
            self.assertEqual(len(adjusted), 1)
            self.assertEqual(adjusted[0]["item_id"], "070_00000001#01")
            # Teacher box [0.10..0.20] vs human [0.30..0.40]: zero overlap.
            self.assertEqual(adjusted[0]["iou_to_teacher"], 0.0)
            untouched = [r for r in seq["items"] if not r["human_adjusted"]]
            self.assertNotIn("iou_to_teacher", untouched[0])


class StoreReplayTest(unittest.TestCase):
    """Journal replay must survive every record shape the server writes."""

    def test_query_edit_preserves_box_across_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(Path(tmp) / "store")
            store.set("070_00000001#01", [0.1, 0.4, 0.2, 0.6], annotator="fang0")
            store.set_query("070_00000001#01", "the gray swan closest to the camera",
                            annotator="fang0")
            # Live in-memory state: the box must survive the query edit...
            self.assertEqual(store.get("070_00000001#01"), [0.1, 0.4, 0.2, 0.6])
            self.assertEqual(store.meta("070_00000001#01")["annotator"], "fang0")
            # ...and a fresh replay of the same journal must too.
            replay = AnnotationStore(Path(tmp) / "store")
            self.assertEqual(replay.get("070_00000001#01"), [0.1, 0.4, 0.2, 0.6])
            self.assertEqual(replay.meta("070_00000001#01")["annotator"], "fang0")
            self.assertEqual(replay.get_query("070_00000001#01"),
                             "the gray swan closest to the camera")

    def test_box_overwrite_and_delete_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(Path(tmp) / "store")
            store.set("f#01", [0.1, 0.4, 0.2, 0.6], annotator="glm-4.6v")
            store.set("f#01", [0.2, 0.4, 0.3, 0.6], annotator="fang0")
            replay = AnnotationStore(Path(tmp) / "store")
            self.assertEqual(replay.get("f#01"), [0.2, 0.4, 0.3, 0.6])
            self.assertEqual(replay.meta("f#01")["annotator"], "fang0")
            store.delete("f#01", annotator="fang0")
            replay2 = AnnotationStore(Path(tmp) / "store")
            self.assertIsNone(replay2.get("f#01"))
            self.assertIsNone(replay2.meta("f#01"))


def make_assembly_manifest(tmp: Path, split: str = "train") -> tuple[Path, Path]:
    """Minimal assembly manifest + dataset index with one frame, two records."""
    sample = "070_00000001" if split == "train" else "004_00000001"
    sequence = sample.split("_")[0]
    index_dir = tmp / "data" / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / f"{split}.json").write_text(
        json.dumps({sample: {"visible": f"Train/{sequence}/color/00000001.png",
                             "bbox": [0.10, 0.40, 0.20, 0.60]}}),
        encoding="utf-8",
    )
    records = [
        {"sequence_id": sequence, "sample_id": sample, "object_index": 1,
         "query": "the white swan on the left side of the image", "bbox": [0.10, 0.40, 0.20, 0.60],
         "source": "real", "category": "swan", "bucket": "spatial"},
        {"sequence_id": sequence, "sample_id": sample, "object_index": 2,
         "query": "a duck closest to the camera", "bbox": [0.50, 0.40, 0.60, 0.60],
         "source": "teacher", "category": "duck", "bucket": "distance"},
    ]
    assembly_path = tmp / f"assembly-{split}.json"
    assembly_path.write_text(
        json.dumps({"metadata": {"run_tag": f"asm-test-{split}", "split": split},
                    "records": records}),
        encoding="utf-8",
    )
    return assembly_path, tmp / "review"


class SeedBatchingTest(unittest.TestCase):
    """Startup bulk seeding must not replay the journal per record."""

    def test_seed_many_journal_snapshot_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(Path(tmp) / "store")
            n = store.seed_many([
                ("070_00000001#01", [0.1, 0.4, 0.2, 0.6], "glm-4.6v"),
                ("070_00000001#02", [0.5, 0.4, 0.6, 0.6], "glm-4.6v"),
            ])
            self.assertEqual(n, 2)
            self.assertEqual(store.get("070_00000001#01"), [0.1, 0.4, 0.2, 0.6])
            self.assertEqual(store.meta("070_00000001#02")["annotator"], "glm-4.6v")
            journal = (Path(tmp) / "store" / "annotations.jsonl").read_text().splitlines()
            self.assertEqual(len(journal), 2)
            snapshot = json.loads(
                (Path(tmp) / "store" / "annotations.predictions.json").read_text()
            )
            self.assertEqual(snapshot, {
                "070_00000001#01": [0.1, 0.4, 0.2, 0.6],
                "070_00000001#02": [0.5, 0.4, 0.6, 0.6],
            })
            # Empty batch: no append, no error.
            self.assertEqual(store.seed_many([]), 0)
            self.assertEqual(len((Path(tmp) / "store" / "annotations.jsonl").read_text().splitlines()), 2)

    def test_assembly_seed_respects_human_and_resyncs_stale_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            assembly_path, review_root = make_assembly_manifest(tmp)
            # Pre-existing store: #01 human-adjusted, #02 a stale teacher seed.
            pre = AnnotationStore(review_root / "asm-test-train")
            pre.set("070_00000001#01", [0.12, 0.42, 0.22, 0.62], annotator="fang0")
            pre.set("070_00000001#02", [0.9, 0.9, 0.95, 0.95], annotator="glm-4.6v")

            session = build_assembly_session(assembly_path, tmp / "data", review_root)
            stats = session["stats"]
            self.assertEqual(stats["seeded"], 1)          # stale teacher re-synced
            self.assertEqual(stats["already_seeded"], 1)  # human box untouched
            store = session["store"]
            self.assertEqual(store.get("070_00000001#01"), [0.12, 0.42, 0.22, 0.62])
            self.assertEqual(store.meta("070_00000001#01")["annotator"], "fang0")
            self.assertEqual(store.get("070_00000001#02"), [0.5, 0.4, 0.6, 0.6])
            journal = (review_root / "asm-test-train" / "annotations.jsonl").read_text().splitlines()
            self.assertEqual(len(journal), 3)  # 2 pre-existing + 1 re-sync record
            items = {i["id"]: i for i in session["items"]}
            self.assertEqual(items["070_00000001#01"]["gt_bbox"], [0.10, 0.40, 0.20, 0.60])
            self.assertIsNone(items["070_00000001#02"]["gt_bbox"])


class MultiCorpusSessionTest(unittest.TestCase):
    """Combined train+val sessions: one server, per-corpus stores."""

    def test_combined_session_tags_and_routes_by_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_path, review_root = make_assembly_manifest(tmp, split="train")
            val_path, _ = make_assembly_manifest(tmp, split="val")
            server, state = create_server(
                census_run_dir=None,
                assembly_path=[train_path, val_path],
                data_root=tmp / "data",
                review_root=review_root,
                host="127.0.0.1",
                port=0,
            )
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                self.assertEqual(state.session["name"].count("+"), 1)
                corpora = [it["corpus"] for it in state.session["items"]]
                self.assertEqual(corpora, ["train", "train", "val", "val"])
                payload = state.session_payload()
                self.assertEqual(payload["total_items"], 4)
                self.assertEqual({e["corpus"] for e in payload["items"]}, {"train", "val"})
                # A query edit on a val item must land in the val store only.
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_address[1]}/api/item/004_00000001%2301/query",
                    data=json.dumps({"query": "a white swan closest to the camera",
                                     "annotator": "fang0"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="PUT",
                )
                with urllib.request.urlopen(request) as resp:
                    self.assertEqual(resp.status, 200)
                val_journal = (review_root / "asm-test-val" / "annotations.jsonl").read_text()
                train_journal = (review_root / "asm-test-train" / "annotations.jsonl").read_text()
                self.assertIn("a white swan closest to the camera", val_journal)
                self.assertNotIn("a white swan closest to the camera", train_journal)
                # Session payload reflects the edit from the routed store.
                fresh = state.session_payload()
                edited = next(e for e in fresh["items"] if e["id"] == "004_00000001#01")
                self.assertEqual(edited["query_en"], "a white swan closest to the camera")
            finally:
                server.shutdown()
                server.server_close()

    def test_duplicate_split_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_a, _ = make_assembly_manifest(tmp, split="train")
            train_b, _ = make_assembly_manifest(tmp, split="train")
            with self.assertRaises(ValueError):
                create_server(
                    census_run_dir=None,
                    assembly_path=[train_a, train_b],
                    data_root=tmp / "data",
                    review_root=tmp / "review",
                    host="127.0.0.1",
                    port=0,
                )


if __name__ == "__main__":
    unittest.main()

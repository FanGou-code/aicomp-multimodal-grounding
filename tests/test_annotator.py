"""Tests for the manifest-driven annotation server (annotator)."""

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

from aicomp_grounding.annotator.server import create_server, SEED_ANNOTATOR
from aicomp_grounding.annotator.store import AnnotationStore


def make_manifest(tmp: Path, split: str = "train") -> Path:
    """Minimal review manifest with one frame and two seeded objects."""
    manifest_path = tmp / f"manifest-{split}.json"
    items = [
        {
            "id": "070_00000001#01",
            "image": "Raw/070/color/00000001.png",
            "query": "the white swan on the left side of the image",
            "bbox": [0.10, 0.40, 0.20, 0.60],
            "corpus": split,
            "source": "real",
            "category": "swan",
            "object_index": 1,
            "frame_id": "070_00000001",
        },
        {
            "id": "070_00000001#02",
            "image": "Raw/070/color/00000001.png",
            "query": "a duck closest to the camera",
            "bbox": [0.50, 0.40, 0.60, 0.60],
            "corpus": split,
            "category": "duck",
            "object_index": 2,
            "frame_id": "070_00000001",
        },
    ]
    manifest_path.write_text(
        json.dumps(
            {"name": f"test-{split}", "run_tag": f"test-{split}", "split": split, "items": items}
        ),
        encoding="utf-8",
    )
    return manifest_path


class ManifestServerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.manifest = make_manifest(self.tmp)
        self.review_root = self.tmp / "review"
        self.server, self.state = create_server(
            data_root=self.tmp / "data",
            review_root=self.review_root,
            manifest_path=self.manifest,
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
        self.assertEqual(payload["total_items"], 2)
        item = payload["items"][0]
        self.assertEqual(item["id"], "070_00000001#01")
        self.assertEqual(item["ordinal"], 1)
        self.assertEqual(item["frame_id"], "070_00000001")
        self.assertEqual(item["gt_bbox"], [0.10, 0.40, 0.20, 0.60])
        self.assertEqual(item["annotator"], SEED_ANNOTATOR)  # seed box seeded
        self.assertAlmostEqual(item["bbox"][0], 0.10)

    def test_restart_resyncs_teacher_but_preserves_human_boxes(self):
        # Human-adjust #01 first; on restart the teacher resync must not
        # clobber it, while #02 (still teacher-seeded) is re-seeded unchanged.
        item_id = "070_00000001#01"
        request = urllib.request.Request(
            f"{self.base_url}/api/item/{urllib.parse.quote(item_id)}/bbox",
            data=json.dumps({"bbox": [0.11, 0.41, 0.21, 0.61], "annotator": "fang0"}).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(request):
            pass
        self.server.server_close()
        server2, state2 = create_server(
            data_root=self.tmp / "data",
            review_root=self.review_root,
            manifest_path=self.manifest,
            host="127.0.0.1",
            port=0,
        )
        threading.Thread(target=server2.serve_forever, daemon=True).start()
        try:
            items = {e["id"]: e for e in state2.session_payload()["items"]}
            self.assertEqual(items[item_id]["bbox"], [0.11, 0.41, 0.21, 0.61])
            self.assertEqual(items[item_id]["annotator"], "fang0")
            # #02 is still teacher-seeded with the manifest box.
            self.assertEqual(items["070_00000001#02"]["annotator"], SEED_ANNOTATOR)
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

    def test_put_bbox_null_marks_absent(self):
        item_id = "070_00000001#01"
        request = urllib.request.Request(
            f"{self.base_url}/api/item/{urllib.parse.quote(item_id)}/bbox",
            data=json.dumps({"bbox": None, "annotator": "fang0:absent"}).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(request) as resp:
            body = json.loads(resp.read())
            self.assertIsNone(body["bbox"])
            self.assertTrue(body.get("absent"))
        payload = self.state.session_payload()
        first = next(e for e in payload["items"] if e["id"] == item_id)
        self.assertIsNone(first["bbox"])
        self.assertEqual(first["annotator"], "fang0:absent")


class QueryLockTest(unittest.TestCase):
    def test_locked_server_rejects_query_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = make_manifest(tmp)
            server, state = create_server(
                data_root=tmp / "data",
                review_root=tmp / "review",
                manifest_path=manifest,
                host="127.0.0.1",
                port=0,
                lock_query=True,
            )
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                request = urllib.request.Request(
                    f"{base}/api/item/070_00000001%2301/query",
                    data=json.dumps({"query": "changed", "annotator": "fang0"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="PUT",
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(request)
                self.assertEqual(ctx.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()


class StoreReplayTest(unittest.TestCase):
    """Journal replay must survive every record shape the server writes."""

    def test_append_during_initial_replay_is_not_marked_as_already_read(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            reader = AnnotationStore(tmp)
            reader.set("a", [0, 0, 1, 1], "reviewer")
            read_state = reader._read_journal_state
            def read_then_append():
                state = read_state()
                AnnotationStore(tmp).set_query("late", "the updated query", "reviewer")
                return state
            with patch.object(reader, "_read_journal_state", side_effect=read_then_append):
                reader._replay()
            self.assertEqual(reader.get_query("late"), "the updated query")

    def test_first_write_after_torn_tail_preserves_original_bytes(self):
        for tail in (b'{"id":"broken"', b'{"query":"\xe4\xb8', b'{"id":"valid","query":"old"}'):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                store = AnnotationStore(directory)
                store.set("a", [0, 0, 1, 1], "reviewer")
                with store.journal_path.open("ab") as handle:
                    handle.write(tail)
                original = store.journal_path.read_bytes()
                recovered = AnnotationStore(directory)
                recovered.set("b", [0.1, 0.2, 0.3, 0.4], "reviewer")
                self.assertEqual(AnnotationStore(directory).get("b"), [0.1, 0.2, 0.3, 0.4])
                self.assertTrue(store.journal_path.read_bytes().startswith(original))

    def test_two_store_instances_serialize_snapshot_publication(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        import aicomp_grounding.annotator.store as module
        with tempfile.TemporaryDirectory() as tmp:
            a, b = AnnotationStore(tmp), AnnotationStore(tmp)
            first_inside, second_started, second_inside, release = (threading.Event() for _ in range(4))
            original = module.atomic_write_json
            def write_snapshot(path, data):
                if path.name == "annotations.predictions.json":
                    if "b" in data:
                        second_inside.set()
                    else:
                        first_inside.set()
                        if not release.wait(5):
                            raise TimeoutError("test failed to release first writer")
                original(path, data)
            def second_write():
                second_started.set()
                b.set("b", [0, 0, 1, 1], "reviewer")
            with patch.object(module, "atomic_write_json", side_effect=write_snapshot), ThreadPoolExecutor(2) as pool:
                first = pool.submit(a.set, "a", [0, 0, 1, 1], "reviewer")
                self.assertTrue(first_inside.wait(5))
                second = pool.submit(second_write)
                try:
                    self.assertTrue(second_started.wait(5))
                    self.assertFalse(second_inside.wait(0.1))
                finally:
                    release.set()
                first.result(timeout=5)
                second.result(timeout=5)
            self.assertEqual(set(AnnotationStore(tmp).all_boxes()), {"a", "b"})
            self.assertEqual(set(json.loads(a.snapshot_path.read_text())), {"a", "b"})

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
            store.set("f#01", [0.1, 0.4, 0.2, 0.6], annotator="seed")
            store.set("f#01", [0.2, 0.4, 0.3, 0.6], annotator="fang0")
            replay = AnnotationStore(Path(tmp) / "store")
            self.assertEqual(replay.get("f#01"), [0.2, 0.4, 0.3, 0.6])
            self.assertEqual(replay.meta("f#01")["annotator"], "fang0")
            store.delete("f#01", annotator="fang0")
            replay2 = AnnotationStore(Path(tmp) / "store")
            self.assertIsNone(replay2.get("f#01"))
            self.assertIsNone(replay2.meta("f#01"))

    def test_stale_process_box_write_keeps_foreign_query_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "store"
            process_a = AnnotationStore(data_dir)
            process_b = AnnotationStore(data_dir)  # created before A's edit
            process_a.set_query("x#01", "the corrected description", annotator="fang0")
            # B never replayed A's edit; a box-only write must not flush B's
            # stale (query-less) state over annotations.queries.json.
            process_b.set("y#01", [0.1, 0.2, 0.3, 0.4], annotator="fang0")
            snapshot = json.loads(
                (data_dir / "annotations.queries.json").read_text(encoding="utf-8")
            )
            self.assertEqual(snapshot.get("x#01"), "the corrected description")
            self.assertEqual(process_b.get_query("x#01"), "the corrected description")


class SeedBatchingTest(unittest.TestCase):
    """Startup bulk seeding must not replay the journal per record."""

    def test_seed_many_journal_snapshot_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(Path(tmp) / "store")
            n = store.seed_many([
                ("070_00000001#01", [0.1, 0.4, 0.2, 0.6], "seed"),
                ("070_00000001#02", [0.5, 0.4, 0.6, 0.6], "seed"),
            ])
            self.assertEqual(n, 2)
            self.assertEqual(store.get("070_00000001#01"), [0.1, 0.4, 0.2, 0.6])
            self.assertEqual(store.meta("070_00000001#02")["annotator"], "seed")
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

    def test_all_queries_and_seed_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(Path(tmp) / "store")
            self.assertEqual(store.all_queries(), {})
            store.set_query("item_1", "query one", annotator="human")
            self.assertEqual(store.all_queries(), {"item_1": "query one"})
            store.seed_queries([("item_2", "query two", None)])
            self.assertEqual(store.all_queries(), {"item_1": "query one", "item_2": "query two"})


class DatasetSessionTest(unittest.TestCase):
    def test_create_server_dataset_mode_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset_path = tmp / "dataset.json"
            dataset_data = {
                "metadata": {"run_tag": "test-dataset-run"},
                "data": {
                    "001_00000001": {
                        "image": "Raw/001/color/00000001.png",
                        "query": "a swan on water",
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    },
                    "001_00000002": {
                        "image": "Raw/001/color/00000002.png",
                        "query": "",
                        "bbox": None,
                    },
                },
            }
            dataset_path.write_text(json.dumps(dataset_data), encoding="utf-8")
            server, state = create_server(
                output_path=dataset_path,
                review_root=tmp / "review",
                host="127.0.0.1",
                port=0,
            )
            try:
                payload = state.session_payload()
                self.assertEqual(payload["total_items"], 2)
                self.assertFalse(state.is_manifest_session)
                self.assertEqual(state.store_for("001_00000001").all_queries(), {"001_00000001": "a swan on water"})
            finally:
                server.server_close()

    def test_approved_dataset_edit_updates_fingerprints(self):
        from aicomp_grounding.contract import (
            ANNOTATION_PROTOCOL_VERSION,
            approved_dataset_fingerprint,
            source_fingerprint,
            validate_approved_artifact,
        )
        from aicomp_grounding.images import trusted_dataset_image_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset_path = tmp / "train.json"
            initial_data = {
                "001_00000001": {
                    "visible": "Raw/001/color/00000001.png",
                    "infrared": "Raw/001/infrared/00000001.png",
                    "depth": "Raw/001/depth/00000001.png",
                    "query": "the red car on the left",
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                    "width": 1920,
                    "height": 1080,
                }
            }
            img_fp = trusted_dataset_image_fingerprint(initial_data, initial_data, require_recorded_size=True)
            initial_artifact = {
                "metadata": {
                    "status": "approved",
                    "protocol_version": ANNOTATION_PROTOCOL_VERSION,
                    "split": "train",
                    "source_fingerprint": source_fingerprint(initial_data),
                    "dataset_fingerprint": approved_dataset_fingerprint(initial_data),
                    "image_fingerprint": img_fp,
                    "sample_count": 1,
                    "sequence_count": 1,
                    "provenance": {"source_type": "human_annotated"},
                },
                "data": initial_data,
            }
            dataset_path.write_text(json.dumps(initial_artifact), encoding="utf-8")
            store = AnnotationStore(
                tmp / "store",
                journal_path=tmp / "journal.jsonl",
                active_dataset_path=dataset_path,
                data_root=tmp,
            )
            store.set("001_00000001", [0.2, 0.2, 0.3, 0.3], annotator="human")
            store.set_query("001_00000001", "the updated car description", annotator="human")

            updated = json.loads(dataset_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["data"]["001_00000001"]["bbox"], [0.2, 0.2, 0.3, 0.3])
            self.assertEqual(updated["data"]["001_00000001"]["query"], "the updated car description")
            self.assertEqual(updated["metadata"]["source_fingerprint"], source_fingerprint(updated["data"]))
            self.assertEqual(updated["metadata"]["dataset_fingerprint"], approved_dataset_fingerprint(updated["data"]))
            validate_approved_artifact(updated, expected_split="train")

    def test_todo_annotator_is_not_counted_as_human_and_leaves_query_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset_path = tmp / "dataset.json"
            dataset_data = {
                "metadata": {"run_tag": "test-todo-run"},
                "data": {
                    "001_00000001": {
                        "visible": "Raw/001/color/00000001.png",
                        "query": "an ambiguous object",
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                },
            }
            dataset_path.write_text(json.dumps(dataset_data), encoding="utf-8")
            server, state = create_server(
                output_path=dataset_path,
                review_root=tmp / "review",
                host="127.0.0.1",
                port=0,
            )
            try:
                store = state.store_for("001_00000001")
                store.set("001_00000001", [0.1, 0.2, 0.3, 0.4], annotator="human:todo")
                payload = state.session_payload()
                self.assertEqual(payload["human_annotated"], 0)
                self.assertEqual(payload["reviewed_frames"], 0)
                synced = json.loads(dataset_path.read_text(encoding="utf-8"))
                self.assertEqual(synced["data"]["001_00000001"]["query"], "")
            finally:
                server.server_close()


class ManifestSessionTest(unittest.TestCase):
    def test_manifest_session_with_mixed_corpuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "manifest.json"
            manifest_data = {
                "name": "mixed-manifest",
                "run_tag": "test-manifest-run",
                "split": "train",
                "items": [
                    {
                        "id": "item_train#01",
                        "image": "Raw/070/color/00000001.png",
                        "query": "train swan",
                        "bbox": [0.1, 0.1, 0.2, 0.2],
                        "corpus": "train",
                    },
                    {
                        "id": "item_val#01",
                        "image": "Raw/004/color/00000001.png",
                        "query": "val swan",
                        "bbox": [0.3, 0.3, 0.4, 0.4],
                        "corpus": "val",
                    },
                ],
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            server, state = create_server(
                manifest_path=manifest_path,
                review_root=tmp / "review",
                host="127.0.0.1",
                port=0,
            )
            try:
                payload = state.session_payload()
                self.assertEqual(payload["total_items"], 2)
                self.assertEqual(payload["manifest"], "mixed-manifest")
                # Ensure store_for routes items to their respective corpus stores
                store_train = state.store_for("item_train#01")
                store_val = state.store_for("item_val#01")
                self.assertIsNotNone(store_train)
                self.assertIsNotNone(store_val)
                self.assertNotEqual(store_train, store_val)
                self.assertEqual(store_train.data_dir.name, "test-manifest-run")
                self.assertEqual(store_val.data_dir.name, "test-manifest-run-val")
            finally:
                server.server_close()

    def test_project_root_constant_points_to_repo_root(self):
        from aicomp_grounding.annotator import server

        expected_root = Path(__file__).resolve().parent.parent
        self.assertEqual(server.PROJECT_ROOT, expected_root)


if __name__ == "__main__":
    unittest.main()

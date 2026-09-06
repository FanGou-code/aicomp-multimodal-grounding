"""Tests for the census review server (foundry.review)."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from foundry.review.server import create_server


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


if __name__ == "__main__":

    unittest.main()

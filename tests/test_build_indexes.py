import json
import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.artifacts import stable_json_hash
from scripts.build_indexes import build_indexes


class BuildIndexesManifestTests(unittest.TestCase):
    def test_manifest_counts_fingerprints_and_scenes_are_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for scene in ("001", "002"):
                sequence = root / "Train" / scene
                sequence.mkdir(parents=True, exist_ok=True)
                (sequence / "groundtruth.txt").write_text(
                    "00000001.png,1,1,3,2\n",
                    encoding="utf-8",
                )

            queries = {
                "000001_001": {
                    "visible": "Images/visible/000001.png",
                    "infrared": "Images/infrared/000001.png",
                    "depth": "Images/depth/000001.png",
                    "query": "The red car",
                }
            }
            queries_path = root / "Test" / "queries" / "queries.json"
            queries_path.parent.mkdir(parents=True, exist_ok=True)
            queries_path.write_text(json.dumps(queries), encoding="utf-8")

            manifest = build_indexes(root, seed=42, train_ratio=0.8)
            train = json.loads((root / "train.json").read_text(encoding="utf-8"))
            val = json.loads((root / "val.json").read_text(encoding="utf-8"))

            self.assertEqual(
                manifest["index_sample_counts"],
                {"train": len(train), "val": len(val)},
            )
            self.assertEqual(manifest["stats"]["train_samples"], len(train))
            self.assertEqual(manifest["stats"]["val_samples"], len(val))
            self.assertEqual(
                manifest["index_fingerprints"]["train"],
                stable_json_hash(train),
            )
            self.assertEqual(
                manifest["index_fingerprints"]["val"],
                stable_json_hash(val),
            )
            self.assertEqual(
                sorted(manifest["train_sequences"]),
                sorted({key.split("_", 1)[0] for key in train}),
            )
            self.assertEqual(
                sorted(manifest["val_sequences"]),
                sorted({key.split("_", 1)[0] for key in val}),
            )


if __name__ == "__main__":
    unittest.main()

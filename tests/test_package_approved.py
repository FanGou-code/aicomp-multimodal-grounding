import json
import tempfile
import unittest
from pathlib import Path

from foundry.pipeline.contract import (
    ANNOTATION_PROTOCOL_VERSION,
    source_fingerprint,
    validate_approved_artifact,
)
from foundry.utils import load_json, stable_json_hash
from scripts.package_approved import package


class PackageApprovedTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.index_data = {
            "001_00000001": {
                "visible": "Train/001/color/00000001.png",
                "infrared": "Train/001/infrared/00000001.png",
                "depth": "Processed/Train/001/depth_jet/00000001.png",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "width": 1920,
                "height": 1080,
            },
            "001_00000002": {
                "visible": "Train/001/color/00000002.png",
                "infrared": "Train/001/infrared/00000002.png",
                "depth": "Processed/Train/001/depth_jet/00000002.png",
                "bbox": [0.2, 0.3, 0.5, 0.6],
                "width": 1920,
                "height": 1080,
            },
        }
        self.index_path = self.root / "train.json"
        self.index_path.write_text(json.dumps(self.index_data), encoding="utf-8")

        self.manifest_data = {
            "status": "complete",
            "preparation_protocol_version": 2,
            "seed": 42,
            "train_ratio": 0.8,
            "depth_scaling": "fixed",
            "min_depth_mm": 300,
            "max_depth_mm": 20000,
            "train_sequences": ["001"],
            "val_sequences": [],
            "stats": {},
            "index_fingerprints": {"train": stable_json_hash(self.index_data), "val": ""},
            "index_sample_counts": {"train": 2, "val": 0},
        }
        self.manifest_path = self.root / "split_manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest_data), encoding="utf-8")

        self.valid_assembly = {
            "metadata": {
                "assembler_version": 1,
                "run_tag": "asm-train-test",
                "census_run_id": "census_test123",
                "census_preparation_fingerprint": stable_json_hash(self.manifest_data),
                "split": "train",
            },
            "records": [
                {
                    "sample_id": "001_00000001",
                    "sequence_id": "001",
                    "object_index": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "query": "The yellow umbrella standing near the doorway",
                },
                {
                    "sample_id": "001_00000002",
                    "sequence_id": "001",
                    "object_index": 2,
                    "bbox": [0.2, 0.3, 0.5, 0.6],
                    "query": "The blue backpack resting on the wooden bench",
                },
            ],
        }
        self.assembly_path = self.root / "assembly.json"
        self.assembly_path.write_text(json.dumps(self.valid_assembly), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_package_approved_valid_output(self):
        out_path = self.root / "approved.json"
        result = package(
            assembly_path=self.assembly_path,
            index_path=self.index_path,
            split_manifest_path=self.manifest_path,
            output_path=out_path,
            run_id="annot_test",
        )
        self.assertTrue(out_path.is_file())
        self.assertEqual(result["run_id"], "annot_test")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["sequence_count"], 1)

        loaded = load_json(out_path)
        validated = validate_approved_artifact(loaded, expected_split="train", expected_run_id="annot_test")
        self.assertEqual(validated["metadata"]["status"], "approved")
        self.assertEqual(validated["metadata"]["protocol_version"], ANNOTATION_PROTOCOL_VERSION)
        self.assertEqual(len(validated["data"]), 2)

    def test_package_approved_deterministic_fingerprints(self):
        out1 = self.root / "out1.json"
        out2 = self.root / "out2.json"
        res1 = package(
            assembly_path=self.assembly_path,
            index_path=self.index_path,
            split_manifest_path=self.manifest_path,
            output_path=out1,
            run_id="annot_det",
        )
        res2 = package(
            assembly_path=self.assembly_path,
            index_path=self.index_path,
            split_manifest_path=self.manifest_path,
            output_path=out2,
            run_id="annot_det",
        )
        self.assertEqual(res1["source_fingerprint"], res2["source_fingerprint"])
        self.assertEqual(res1["dataset_fingerprint"], res2["dataset_fingerprint"])
        self.assertEqual(res1["image_fingerprint"], res2["image_fingerprint"])
        self.assertEqual(res1["preparation_fingerprint"], res2["preparation_fingerprint"])
        self.assertEqual(out1.read_text(encoding="utf-8"), out2.read_text(encoding="utf-8"))

    def test_package_approved_query_qc_rejection_and_lenient(self):
        bad_assembly = {
            "metadata": {
                "assembler_version": 1,
                "run_tag": "asm-bad",
                "census_run_id": "c1",
                "census_preparation_fingerprint": "p1",
                "split": "train",
            },
            "records": [
                {
                    "sample_id": "001_00000001",
                    "sequence_id": "001",
                    "object_index": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "query": "The person on the left side of the image",  # contains 'image'
                }
            ],
        }
        bad_path = self.root / "bad_assembly.json"
        bad_path.write_text(json.dumps(bad_assembly), encoding="utf-8")

        # Strict QC should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            package(
                assembly_path=bad_path,
                index_path=self.index_path,
                split_manifest_path=self.manifest_path,
                output_path=self.root / "never.json",
                lenient_qc=False,
            )
        self.assertIn("Query QC validation failed", str(ctx.exception))

        # Lenient QC should succeed with warning and write output
        lenient_out = self.root / "lenient.json"
        res = package(
            assembly_path=bad_path,
            index_path=self.index_path,
            split_manifest_path=self.manifest_path,
            output_path=lenient_out,
            lenient_qc=True,
        )
        self.assertTrue(lenient_out.is_file())
        self.assertEqual(res["qc_failures_count"], 1)

    def test_package_approved_export_to_main(self):
        main_mock = self.root / "main_repo"
        package(
            assembly_path=self.assembly_path,
            index_path=self.index_path,
            split_manifest_path=self.manifest_path,
            output_path=self.root / "approved.json",
            export_to_main=main_mock,
            run_id="annot_export_test",
        )
        exported = main_mock / "outputs" / "annotations" / "annot_export_test" / "train" / "approved.json"
        self.assertTrue(exported.is_file())
        loaded = load_json(exported)
        self.assertEqual(loaded["metadata"]["run_id"], "annot_export_test")

    def test_package_approved_cross_repo_compatibility(self):
        """Verify the packaged artifact passes the main repo's validator directly."""
        main_repo_path = Path("/home/fang0/dev/projects/aicomp-multimodal-grounding")
        if not main_repo_path.is_dir():
            self.skipTest("Main repo not available on filesystem")

        import sys
        if str(main_repo_path) not in sys.path:
            sys.path.insert(0, str(main_repo_path))

        try:
            from aicomp_grounding.annotation_state import (
                validate_approved_artifact as main_repo_validate,
            )
        except ImportError:
            self.skipTest("Could not import aicomp_grounding")

        out_path = self.root / "approved.json"
        res = package(
            assembly_path=self.assembly_path,
            index_path=self.index_path,
            split_manifest_path=self.manifest_path,
            output_path=out_path,
            run_id="annot_cross_test",
        )
        loaded = load_json(out_path)
        main_validated = main_repo_validate(
            loaded, expected_split="train", expected_run_id="annot_cross_test"
        )
        self.assertEqual(main_validated["metadata"]["run_id"], "annot_cross_test")
        self.assertEqual(main_validated["metadata"]["dataset_fingerprint"], res["dataset_fingerprint"])


if __name__ == "__main__":
    unittest.main()

"""Offline behavior tests for compliant sequence-level annotation artifacts."""

from __future__ import annotations

import copy
import json
import unittest

from aicomp_grounding.annotation_state import (
    ANNOTATION_MODE,
    ASSIGNMENT_POLICY,
    annotation_attempt_numbers,
    approved_dataset_fingerprint,
    build_annotation_plan,
    build_annotation_shard_metadata,
    build_approved_artifact,
    merge_annotation_payloads,
    pending_frames,
    pending_sequences,
    validate_annotation_checkpoint,
    validate_approved_artifact,
)
from aicomp_grounding.config import ANNOTATION_PROTOCOL_VERSION
from aicomp_grounding.sequence import source_fingerprint
from aicomp_grounding.sharding import group_keys_by_scene


def _dataset(sequence_count: int = 6, frames: int = 4, old_query: str = "closed API label") -> dict:
    result = {}
    for scene in range(1, sequence_count + 1):
        scene_id = f"{scene:03d}"
        for frame in range(1, frames + 1):
            sample_id = f"{scene_id}_{frame:08d}"
            result[sample_id] = {
                "visible": f"Train/{scene_id}/color/{frame:08d}.png",
                "infrared": f"Train/{scene_id}/infrared/{frame:08d}.png",
                "depth": f"Processed/Train/{scene_id}/depth_jet/{frame:08d}.png",
                "query": old_query,
                "bbox": [0.1, 0.1, 0.3 + frame * 0.01, 0.4],
                "width": 1920,
                "height": 1080,
                "internal": "must not publish",
            }
    return result


def _plan(data: dict, *, limit=None, shards=2) -> dict:
    return build_annotation_plan(
        data,
        split="train",
        provider="zhipu",
        api_base_url="https://api.siliconflow.cn/v1",
        model_name="open-model",
        model_revision="revision",
        model_weights_url="https://example.com/open-model",
        model_license="Apache-2.0",
        prompt_hash="prompt",
        render_protocol="clean-views-v1",
        generation_config={"max_tokens": 256, "enable_thinking": False},
        preparation_fingerprint="preparation-bytes",
        image_fingerprint="image-bytes",
        seed=42,
        limit_sequences=limit,
        requested_num_shards=shards,
        run_tag="round1",
    )


def _completed_result(sequence_id: str, data: dict) -> dict:
    keys = group_keys_by_scene(list(data), data)[sequence_id]
    call = {
        "response_id": "response",
        "trace_id": "trace",
        "model": "open-model",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
    }
    frames = {}
    for sample_id in keys:
        frames[sample_id] = {
            "status": "completed",
            "query": "The pedestrian wearing a yellow coat with dark trousers",
            "uncertain": False,
            "attempts": 1,
            "error": "",
            "api_calls": [call],
        }
    return {
        "status": "completed",
        "frames": frames,
    }


class PlanTests(unittest.TestCase):
    def test_legacy_source_queries_do_not_affect_identity(self):
        first = _dataset(old_query="first closed label")
        second = _dataset(old_query="different closed label")
        plan_a = _plan(first)
        plan_b = _plan(second)
        self.assertEqual(plan_a["metadata"]["source_fingerprint"], plan_b["metadata"]["source_fingerprint"])
        self.assertEqual(plan_a["metadata"]["run_id"], plan_b["metadata"]["run_id"])

    def test_limit_is_global_and_does_not_create_empty_shards(self):
        data = _dataset(sequence_count=8)
        plan = _plan(data, limit=5, shards=4)
        self.assertEqual(sum(len(shard) for shard in plan["shards"]), 5)
        self.assertEqual(len(plan["shards"]), 4)
        self.assertTrue(all(plan["shards"]))
        tiny = _plan(data, limit=2, shards=10)
        self.assertEqual(len(tiny["shards"]), 2)

    def test_shard_count_does_not_change_cross_split_campaign_identity(self):
        data = _dataset(sequence_count=8)
        one_shard = _plan(data, shards=1)
        four_shards = _plan(data, shards=4)
        self.assertEqual(
            one_shard["metadata"]["run_id"],
            four_shards["metadata"]["run_id"],
        )
        self.assertNotEqual(
            one_shard["metadata"]["requested_num_shards"],
            four_shards["metadata"]["requested_num_shards"],
        )

    def test_preparation_changes_campaign_and_image_bytes_change_split_plan(self):
        data = _dataset(sequence_count=2)
        first = _plan(data)
        changed_preparation = build_annotation_plan(
            data,
            split="train",
            provider="zhipu",
            api_base_url="https://api.siliconflow.cn/v1",
            model_name="open-model",
            model_revision="revision",
            model_weights_url="https://example.com/open-model",
            model_license="Apache-2.0",
            prompt_hash="prompt",
            render_protocol="clean-views-v1",
            generation_config={"max_tokens": 256, "enable_thinking": False},
            preparation_fingerprint="different-preparation",
            image_fingerprint="image-bytes",
            seed=42,
            limit_sequences=None,
            requested_num_shards=2,
            run_tag="round1",
        )
        changed_images = build_annotation_plan(
            data,
            split="train",
            provider="zhipu",
            api_base_url="https://api.siliconflow.cn/v1",
            model_name="open-model",
            model_revision="revision",
            model_weights_url="https://example.com/open-model",
            model_license="Apache-2.0",
            prompt_hash="prompt",
            render_protocol="clean-views-v1",
            generation_config={"max_tokens": 256, "enable_thinking": False},
            preparation_fingerprint="preparation-bytes",
            image_fingerprint="different-images",
            seed=42,
            limit_sequences=None,
            requested_num_shards=2,
            run_tag="round1",
        )
        self.assertNotEqual(first["metadata"]["run_id"], changed_preparation["metadata"]["run_id"])
        self.assertEqual(first["metadata"]["run_id"], changed_images["metadata"]["run_id"])
        self.assertNotEqual(first["metadata"], changed_images["metadata"])


class CheckpointAndMergeTests(unittest.TestCase):
    def test_checkpoint_requires_every_metadata_field(self):
        data = _dataset(sequence_count=1)
        plan = _plan(data, shards=1)
        assigned = plan["shards"][0]
        metadata = build_annotation_shard_metadata(plan["metadata"], 0, assigned, data)
        result = {assigned[0]: _completed_result(assigned[0], data)}
        payload = {"metadata": metadata, "results": result}
        validate_annotation_checkpoint(payload, metadata, assigned, data, require_complete=True)
        broken = copy.deepcopy(payload)
        del broken["metadata"]["seed"]
        with self.assertRaises(ValueError):
            validate_annotation_checkpoint(broken, metadata, assigned, data, require_complete=True)

    def test_failed_sequences_only_retry_when_explicit(self):
        assigned = ["001", "002", "003"]
        results = {
            "001": {"status": "completed"},
            "002": {"status": "failed"},
        }
        self.assertEqual(pending_sequences(assigned, results, retry_failed=False), ["003"])
        self.assertEqual(pending_sequences(assigned, results, retry_failed=True), ["002", "003"])
        frames = {
            "001_1": {"status": "completed"},
            "001_2": {"status": "failed"},
        }
        sample_ids = ["001_1", "001_2", "001_3"]
        self.assertEqual(pending_frames(sample_ids, frames, retry_failed=False), ["001_3"])
        self.assertEqual(
            pending_frames(sample_ids, frames, retry_failed=True),
            ["001_2", "001_3"],
        )
        self.assertEqual(
            annotation_attempt_numbers(None, retry_failed=True),
            [1, 2, 3],
        )
        failed_with_history = {"status": "failed", "attempts": 3, "error": "bad JSON"}
        self.assertEqual(
            annotation_attempt_numbers(failed_with_history, retry_failed=True),
            [4, 5, 6],
        )

    def test_checkpoint_attempt_counts_reject_booleans(self):
        data = _dataset(sequence_count=1)
        plan = _plan(data, shards=1)
        assigned = plan["shards"][0]
        metadata = build_annotation_shard_metadata(plan["metadata"], 0, assigned, data)

        failed = _completed_result(assigned[0], data)
        failed["frames"][next(iter(failed["frames"]))]["attempts"] = True
        with self.assertRaisesRegex(ValueError, "attempt count"):
            validate_annotation_checkpoint(
                {"metadata": metadata, "results": {assigned[0]: failed}},
                metadata,
                assigned,
                data,
                require_complete=True,
            )

    def test_completed_frame_rejects_invalid_query(self):
        data = _dataset(sequence_count=1)
        plan = _plan(data, shards=1)
        assigned = plan["shards"][0]
        metadata = build_annotation_shard_metadata(plan["metadata"], 0, assigned, data)
        completed = _completed_result(assigned[0], data)
        frame = next(iter(completed["frames"].values()))
        frame["query"] = "object"
        with self.assertRaisesRegex(ValueError, "query is invalid"):
            validate_annotation_checkpoint(
                {"metadata": metadata, "results": {assigned[0]: completed}},
                metadata,
                assigned,
                data,
                require_complete=True,
            )

    def test_merge_rejects_missing_shard(self):
        data = _dataset(sequence_count=2)
        plan = _plan(data, shards=2)
        payloads = []
        for shard_id, assigned in enumerate(plan["shards"]):
            metadata = build_annotation_shard_metadata(plan["metadata"], shard_id, assigned, data)
            payloads.append({
                "metadata": metadata,
                "results": {sequence_id: _completed_result(sequence_id, data) for sequence_id in assigned},
            })
        merged = merge_annotation_payloads(plan, list(reversed(payloads)), data)
        self.assertEqual(set(merged), {"001", "002"})
        with self.assertRaises(ValueError):
            merge_annotation_payloads(plan, payloads[:1], data)


class ApprovalTests(unittest.TestCase):
    def test_approved_artifact_is_clean_complete_and_open_weights_hosted(self):
        data = _dataset(sequence_count=2)
        plan = _plan(data, shards=2)
        results = {sequence_id: _completed_result(sequence_id, data) for sequence_id in ("001", "002")}
        artifact = build_approved_artifact(data, plan, results)
        validated = validate_approved_artifact(
            artifact,
            expected_split="train",
            expected_run_id=plan["metadata"]["run_id"],
        )
        self.assertIs(validated, artifact)
        provenance = artifact["metadata"]["provenance"]
        self.assertEqual(provenance["source_type"], "hosted_open_weights")
        self.assertEqual(provenance["provider"], "zhipu")
        self.assertEqual(provenance["model_license"], "Apache-2.0")
        for item in artifact["data"].values():
            self.assertNotIn("internal", item)
            self.assertNotIn("completed", item)
            self.assertNotEqual(item["query"], "closed API label")

    def test_limited_or_failed_run_cannot_be_approved(self):
        data = _dataset(sequence_count=2)
        limited = _plan(data, limit=1, shards=1)
        selected = limited["metadata"]["selected_sequence_ids"][0]
        with self.assertRaises(ValueError):
            build_approved_artifact(data, limited, {selected: _completed_result(selected, data)})

        full = _plan(data, shards=2)
        results = {
            "001": _completed_result("001", data),
            "002": {
                "status": "failed",
                "frames": {},
            },
        }
        with self.assertRaises(ValueError):
            build_approved_artifact(data, full, results)

    def test_tampered_provenance_or_dataset_is_rejected(self):
        data = _dataset(sequence_count=1)
        plan = _plan(data, shards=1)
        artifact = build_approved_artifact(data, plan, {"001": _completed_result("001", data)})
        tampered = copy.deepcopy(artifact)
        tampered["metadata"]["provenance"]["source_type"] = "closed_api"
        with self.assertRaises(ValueError):
            validate_approved_artifact(tampered, expected_split="train", expected_run_id=plan["metadata"]["run_id"])
        tampered = copy.deepcopy(artifact)
        next(iter(tampered["data"].values()))["query"] = "Changed after approval"
        with self.assertRaises(ValueError):
            validate_approved_artifact(tampered, expected_split="train", expected_run_id=plan["metadata"]["run_id"])

    def test_empty_approved_artifact_is_rejected(self):
        artifact = {
            "metadata": {
                "status": "approved",
                "protocol_version": ANNOTATION_PROTOCOL_VERSION,
                "run_id": "annot_empty",
                "split": "train",
                "source_fingerprint": source_fingerprint({}),
                "preparation_fingerprint": "preparation-bytes",
                "image_fingerprint": "image-bytes",
                "dataset_fingerprint": approved_dataset_fingerprint({}),
                "sample_count": 0,
                "sequence_count": 0,
                "prompt_hash": "prompt",
                "provenance": {
                    "source_type": "hosted_open_weights",
                    "provider": "zhipu",
                    "api_base_url": "https://api.zhipu.cn/v1",
                    "annotator_model": "open-model",
                    "annotator_revision": "revision",
                    "model_weights_url": "https://example.com/open-model",
                    "model_license": "Apache-2.0",
                    "mode": ANNOTATION_MODE,
                    "assignment_policy": ASSIGNMENT_POLICY,
                    "render_protocol": "clean-views-v1",
                    "generation_config": {"max_tokens": 256, "enable_thinking": False},
                },
                "qc": {
                    "complete": True,
                    "failed_sequences": 0,
                    "failed_frames": 0,
                    "invalid_queries": 0,
                    "generated_samples": 0,
                },
            },
            "data": {},
        }
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_approved_artifact(
                artifact,
                expected_split="train",
                expected_run_id="annot_empty",
            )


if __name__ == "__main__":
    unittest.main()

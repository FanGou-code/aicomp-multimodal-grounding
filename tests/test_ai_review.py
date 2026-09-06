"""Offline tests for the AI pre-review pass (no API)."""

import json
import tempfile
import unittest
from pathlib import Path

from foundry.review.store import AnnotationStore
from scripts.run_ai_review import (
    AI_REVIEW_PROMPT,
    VALID_VERDICTS,
    audit_messages,
    build_plan,
    item_id_of,
    normalize_query,
    parse_audit_response,
    prompt_hash,
)


def _payload(verdict="pass", query="the white swan closest to the camera",
             observed="3 swans; red box is nearest", reason="ok"):
    return json.dumps({
        "observed": observed, "verdict": verdict, "query": query, "reason": reason,
    })


def _assembly_fixture(tmp: Path, run_tag: str = "asm-test", split: str = "train",
                      records=None) -> Path:
    sample = "070_00000001" if split == "train" else "004_00000001"
    sequence = sample.split("_")[0]
    index_dir = tmp / "data" / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / f"{split}.json").write_text(json.dumps({
        sample: {"visible": f"Train/{sequence}/color/00000001.png",
                 "bbox": [0.10, 0.40, 0.20, 0.60]},
    }), encoding="utf-8")
    records = records or [
        {"sequence_id": sequence, "sample_id": sample, "object_index": 1,
         "query": "the white swan closest to the camera", "bbox": [0.10, 0.40, 0.20, 0.60],
         "source": "real", "category": "swan", "bucket": "distance"},
        {"sequence_id": sequence, "sample_id": sample, "object_index": 2,
         "query": "a duck farthest from the camera", "bbox": [0.50, 0.40, 0.60, 0.60],
         "source": "teacher", "category": "duck", "bucket": "distance"},
    ]
    path = tmp / f"assembly-{run_tag}.json"
    path.write_text(json.dumps({
        "metadata": {"run_tag": run_tag, "split": split}, "records": records,
    }), encoding="utf-8")
    return path


class ParseAuditResponseTest(unittest.TestCase):
    def test_pass_parsed_verbatim(self):
        parsed = parse_audit_response(_payload(), "the white swan closest to the camera")
        self.assertEqual(parsed["verdict"], "pass")
        self.assertEqual(parsed["query"], "the white swan closest to the camera")
        self.assertFalse(parsed["coerced"])

    def test_fixed_kept(self):
        parsed = parse_audit_response(
            _payload(verdict="fixed", query="the second white swan from the left"),
            "the white swan closest to the camera",
        )
        self.assertEqual(parsed["verdict"], "fixed")
        self.assertFalse(parsed["coerced"])

    def test_pass_with_changed_query_coerced_to_fixed(self):
        parsed = parse_audit_response(
            _payload(verdict="pass", query="the second white swan from the left"),
            "the white swan closest to the camera",
        )
        self.assertEqual(parsed["verdict"], "fixed")
        self.assertTrue(parsed["coerced"])

    def test_fixed_with_same_query_coerced_to_pass(self):
        parsed = parse_audit_response(
            _payload(verdict="fixed"), "the white swan closest to the camera"
        )
        self.assertEqual(parsed["verdict"], "pass")
        self.assertTrue(parsed["coerced"])

    def test_invalid_verdict_rejected(self):
        with self.assertRaises(ValueError):
            parse_audit_response(_payload(verdict="maybe"), "q")

    def test_schema_and_duplicate_keys_rejected(self):
        bad_schema = json.dumps({"observed": "x", "verdict": "pass", "query": "q"})
        with self.assertRaises(ValueError):
            parse_audit_response(bad_schema, "q")
        with self.assertRaises(ValueError):
            parse_audit_response(
                '{"observed":"x","observed":"y","verdict":"pass","query":"q","reason":"r"}', "q"
            )

    def test_query_normalized(self):
        parsed = parse_audit_response(
            _payload(query="  the white   swan closest to the camera. "),
            "the white swan closest to the camera",
        )
        self.assertEqual(parsed["query"], "the white swan closest to the camera")

    def test_normalize_query_strips_terminal_punctuation(self):
        self.assertEqual(normalize_query("the white swan."), "the white swan")


class BuildPlanTest(unittest.TestCase):
    def test_plan_skips_human_reviewed_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            assembly = _assembly_fixture(tmp)
            review_root = tmp / "review"
            # Item #01 carries a human verdict; #02 is still teacher-seeded.
            store = AnnotationStore(review_root / "asm-test")
            store.set("070_00000001#01", [0.1, 0.4, 0.2, 0.6], annotator="fang0")
            store.set("070_00000001#02", [0.5, 0.4, 0.6, 0.6], annotator="glm-4.6v")

            plan = build_plan(assembly, tmp / "data", review_root)
            self.assertEqual(plan["run_tag"], "asm-test")
            self.assertEqual(plan["skipped_human"], 1)
            self.assertEqual([it["item_id"] for it in plan["items"]], ["070_00000001#02"])
            self.assertEqual(plan["records_total"], 2)

    def test_plan_without_review_store_includes_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            assembly = _assembly_fixture(tmp)
            plan = build_plan(assembly, tmp / "data", tmp / "review")
            self.assertEqual(len(plan["items"]), 2)
            self.assertEqual(plan["skipped_human"], 0)


class PromptContractTest(unittest.TestCase):
    """The frozen audit contract: restricted edits, three verdicts, re-check."""

    def test_prompt_pins_the_contract(self):
        for needle in (
            "thick red rectangle",
            "left to right <-> right to left",
            "A correction is an edit, never a rewrite",
            "RE-CHECK YOUR EDIT",
            '"pass" | "fixed" | "human"',
            "Never restructure the sentence",
        ):
            self.assertIn(needle, AI_REVIEW_PROMPT)

    def test_prompt_embeds_query_via_placeholder(self):
        self.assertIn("<query>", AI_REVIEW_PROMPT)
        messages = audit_messages("data:image/jpeg;base64,xxx", "the gray swan")
        text_parts = [p.get("text", "") for p in messages[1]["content"] if "text" in p]
        self.assertIn("the gray swan", "".join(text_parts))

    def test_verdicts_and_hash_stable(self):
        self.assertEqual(VALID_VERDICTS, ("pass", "fixed", "human"))
        self.assertEqual(prompt_hash(), prompt_hash())
        self.assertEqual(item_id_of({"sample_id": "070_00000001", "object_index": 3}),
                         "070_00000001#03")


if __name__ == "__main__":
    unittest.main()

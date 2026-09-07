"""Tests for the deterministic text QC stage (foundry.text_qc)."""

import unittest

from foundry.pipeline.assembly import AssemblyRecord
from foundry.pipeline.text_qc import adjudicate, apply_text_qc, qc_query


def record(sample_id: str, object_index: int, query: str) -> AssemblyRecord:
    return AssemblyRecord(
        sample_id=sample_id, sequence_id=sample_id.split("_")[0], source="teacher",
        category="swan", bbox=[0.1, 0.4, 0.2, 0.6], object_index=object_index,
        query=query, family="plain_attribute", bucket="attribute_action",
        quota_state="quota", facts=["feature"], words=len(query.split()),
    )


class ArticleEngineTest(unittest.TestCase):
    def test_article_inserted_for_countable_tail(self):
        self.assertEqual(qc_query("The person with small car"),
                         ("The person with a small car", "small car"))
        self.assertEqual(qc_query("The bird with head down"),
                         ("The bird with its head down", "head down"))

    def test_an_before_vowel_letter(self):
        self.assertEqual(adjudicate("office chair"), ("WITH", "an office chair"))

    def test_keep_and_mass_tails_unchanged(self):
        self.assertIsNone(qc_query("The deer with long tail feathers"))
        self.assertIsNone(qc_query("The yard with grass"))
        self.assertIsNone(qc_query("The person with dark pants on person"))

    def test_already_articled_tail_untouched(self):
        self.assertIsNone(qc_query("The person with a small car"))
        self.assertIsNone(qc_query("The swan with the curved neck"))

    def test_wearing_tail_gets_article(self):
        self.assertEqual(qc_query("The person wearing hat"),
                         ("The person wearing a hat", "hat"))


class ApplyTextQcTest(unittest.TestCase):
    def test_engine_edit_sets_edited_and_logs(self):
        records = [record("006_00000129", 3, "The person with small car")]
        edits = apply_text_qc(records)
        self.assertTrue(records[0].edited)
        self.assertEqual(records[0].query, "The person with a small car")
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["item_id"], "006_00000129#03")
        self.assertEqual(edits[0]["before"], "The person with small car")
        self.assertEqual(edits[0]["after"], "The person with a small car")
        self.assertIn("article/echo adjudication", edits[0]["reason"])

    def test_clean_record_untouched(self):
        records = [record("006_00000129", 3, "The person with a small car")]
        edits = apply_text_qc(records)
        self.assertFalse(records[0].edited)
        self.assertEqual(edits, [])

    def test_echo_ruling_applies_by_item_and_before(self):
        records = [record("028_00000026", 4, "The deer lying deer in dirt bed")]
        edits = apply_text_qc(records)
        # The engine does not fire (no with/wearing/holding/carrying tail);
        # the frozen echo table carries the ruling.
        self.assertEqual(records[0].query, "The deer lying in a dirt bed")
        self.assertTrue(records[0].edited)
        self.assertEqual(edits[0]["reason"], "head-echo adjudication")

    def test_echo_ruling_requires_matching_item(self):
        # Same before text but a different item id: the ruling must not fire.
        records = [record("999_00000001", 1, "The deer lying deer in dirt bed")]
        edits = apply_text_qc(records)
        self.assertEqual(records[0].query, "The deer lying deer in dirt bed")
        self.assertEqual(edits, [])

    def test_words_field_stays_pre_qc(self):
        # Historical behaviour: the audit word counts are pre-QC.
        records = [record("006_00000129", 3, "The person with small car")]
        apply_text_qc(records)
        self.assertEqual(records[0].words, 5)


if __name__ == "__main__":
    unittest.main()

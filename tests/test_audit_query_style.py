import unittest

from scripts.audit_query_style import (
    extract_queries,
    format_stats_row,
    style_stats,
)


class ExtractQueriesTests(unittest.TestCase):
    def test_approved_artifact_with_data_dict(self):
        payload = {
            "metadata": {"run_id": "annot_x"},
            "data": {
                "001_1": {"query": "The red car on the left"},
                "001_2": {"query": "A bird"},
            },
        }
        self.assertEqual(extract_queries(payload), ["The red car on the left", "A bird"])

    def test_payload_mapping_with_metadata_fields(self):
        payload = {
            "000001_001": {
                "visible": "Images/visible/000001.png",
                "query": "The nearest tree on the right",
                "bbox": [0.1, 0.2, 0.3, 0.4],
            }
        }
        self.assertEqual(extract_queries(payload), ["The nearest tree on the right"])

    def test_list_of_entries(self):
        payload = [{"query": "The leftmost cone"}, {"query": "The gray rock"}]
        self.assertEqual(extract_queries(payload), ["The leftmost cone", "The gray rock"])

    def test_no_queries_raises_or_returns_empty(self):
        self.assertEqual(extract_queries({"metadata": {}}), [])
        self.assertEqual(extract_queries("not a payload"), [])


class StyleStatsTests(unittest.TestCase):
    def test_mixed_queries(self):
        # Note: word-based counting is an approximation — "top" in
        # "pink top" (clothing) also matches the spatial word list.
        queries = [
            "The third traffic cone from left to right in the front row",
            "The deer in the middle of the image",
            "The person wearing a pink top bending over",
            "A bus",
        ]
        stats = style_stats(queries)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["mean_words"], 7.5)
        self.assertEqual(stats["spatial_ratio"], 0.75)
        self.assertEqual(stats["ordinal_ratio"], 0.25)
        self.assertEqual(stats["spatial_or_ordinal_ratio"], 0.75)
        self.assertEqual(stats["starts_with_the_ratio"], 0.75)

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            style_stats([])

    def test_format_row_includes_key_fields(self):
        stats = style_stats(["The red car on the left", "A bird"])
        row = format_stats_row("sample", stats)
        self.assertIn("sample", row)
        self.assertIn("spatial=", row)
        self.assertIn("either=", row)


if __name__ == "__main__":
    unittest.main()

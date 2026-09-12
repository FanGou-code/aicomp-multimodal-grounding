import math
import unittest

from aicomp_grounding.bbox import (
    compute_iou,
    format_qwen_bbox,
    normalize_pixel_bbox,
    parse_bbox_from_text,
    validate_bbox,
)


class BBoxTests(unittest.TestCase):
    def test_quantized_edge_boxes_stay_in_range_and_nonempty(self):
        for box in ([0.9996, 0.1, 1, 0.5], [0.1, 0.9996, 0.5, 1], [0, 0, 0.0001, 0.0001]):
            with self.subTest(box=box):
                parsed = parse_bbox_from_text(format_qwen_bbox(box))
                self.assertIsNotNone(parsed)
                self.assertTrue(all(0 <= v <= 1 for v in parsed))
                self.assertLess(parsed[0], parsed[2])
                self.assertLess(parsed[1], parsed[3])

    def test_qwen_round_trip_preserves_xy_order(self):
        box = [0.1, 0.2, 0.3, 0.4]
        encoded = format_qwen_bbox(box)
        self.assertEqual(encoded, "<|box_start|>(100,200),(300,400)<|box_end|>")
        self.assertEqual(parse_bbox_from_text(encoded), box)

    def test_small_qwen_coordinates_are_not_mistaken_for_normalized(self):
        self.assertEqual(parse_bbox_from_text("(0,0),(1,1)"), [0.0, 0.0, 0.001, 0.001])

    def test_parser_rejects_prose_numbers_and_negative_coordinates(self):
        self.assertIsNone(parse_bbox_from_text("Image 1 result: (100,200),(300,400)"))
        self.assertIsNone(parse_bbox_from_text("(-10,20),(300,400)"))

    def test_parser_accepts_normalized_values_only_when_requested(self):
        self.assertEqual(
            parse_bbox_from_text("[0.1,0.2,0.3,0.4]", coordinate_scale="normalized"),
            [0.1, 0.2, 0.3, 0.4],
        )

    def test_parser_accepts_explicit_json_and_special_token_wrappers(self):
        self.assertEqual(
            parse_bbox_from_text('{"bbox_2d": [100, 200, 300, 400]}'),
            [0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual(
            parse_bbox_from_text("Result: <|box_start|>(100,200),(300,400)<|box_end|>"),
            [0.1, 0.2, 0.3, 0.4],
        )

    def test_parser_rejects_ambiguous_embedded_boxes(self):
        self.assertIsNone(
            parse_bbox_from_text(
                "<|box_start|>(100,200),(300,400)<|box_end|> "
                "<|box_start|>(500,600),(700,800)<|box_end|>"
            )
        )

    def test_validation_rejects_invalid_boxes(self):
        self.assertIsNone(validate_bbox(None))
        self.assertIsNone(validate_bbox([False, 0.2, 0.3, 0.4]))
        self.assertIsNone(validate_bbox([0.5, 0.2, 0.1, 0.4]))
        self.assertIsNone(validate_bbox([0.1, 0.2, math.nan, 0.4]))
        self.assertIsNone(validate_bbox([-0.1, 0.2, 0.3, 0.4]))

    def test_pixel_normalization(self):
        self.assertEqual(
            normalize_pixel_bbox(570, 578, 305, 73, 1920, 1080),
            [0.296875, 0.535185, 0.455729, 0.602778],
        )

    def test_iou(self):
        self.assertEqual(compute_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(
            compute_iou([0, 0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]),
            1 / 7,
        )
        self.assertEqual(compute_iou([0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()

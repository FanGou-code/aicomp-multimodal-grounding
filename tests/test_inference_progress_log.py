import unittest

from tools.infer import _format_eta, _now_str


class InferenceProgressLogTests(unittest.TestCase):
    def test_eta_formats_hours_minutes_seconds(self):
        self.assertEqual(_format_eta(45), "45s")
        self.assertEqual(_format_eta(90), "1m30s")
        self.assertEqual(_format_eta(3600), "1h00m")
        self.assertEqual(_format_eta(-1), "-")

    def test_timestamp_is_absolute_and_second_precision(self):
        value = _now_str()
        self.assertRegex(value, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


if __name__ == "__main__":
    unittest.main()

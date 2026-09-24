"""Tests for the admin key probe's HTTP status classification."""

from __future__ import annotations

import unittest

from tools.check_key import VERDICTS, classify_status


class ClassifyStatusTests(unittest.TestCase):
    def test_known_statuses_use_their_verdict(self):
        for status, verdict in VERDICTS.items():
            with self.subTest(status=status):
                self.assertEqual(classify_status(status, ""), verdict)

    def test_provider_5xx_is_alive(self):
        for status in (500, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(classify_status(status, "provider exploded").startswith("FAKE-DEAD"))

    def test_transport_failure_keeps_the_detail(self):
        self.assertEqual(classify_status(None, ""), "FAKE-DEAD (transport)")
        self.assertIn("timeout", classify_status(None, "timeout after 120s"))

    def test_unexpected_status_is_named(self):
        self.assertEqual(classify_status(418, ""), "UNEXPECTED HTTP 418")


if __name__ == "__main__":
    unittest.main()

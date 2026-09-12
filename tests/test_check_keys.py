"""Offline tests for the key health prober (no API calls)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_keys import classify_status, read_key_lines


class ReadKeyLinesTests(unittest.TestCase):
    def test_reads_all_lines_including_commented(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_keys.txt"
            path.write_text(
                "key-one\n"
                "\n"
                "# 2026-09-06 quota exhausted\n"
                "#key-two\n"
                "key-three # inline note\n",
                encoding="utf-8",
            )
            entries = read_key_lines(path)
        self.assertEqual(entries, [
            (1, "key-one", False),
            (4, "key-two", True),
            (5, "key-three", False),
        ])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_key_lines(Path("/nonexistent/keys.txt"))


class ClassifyStatusTests(unittest.TestCase):
    def test_retest_recovery_produces_success_summary(self):
        import contextlib
        import io
        from unittest.mock import patch
        from scripts import check_keys
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake-keys.txt"
            path.write_text("test-placeholder\n")
            with patch("sys.argv", ["check_keys.py", "--file", str(path), "--probe", "minimal", "--pause", "1"]), \
                 patch.object(check_keys, "probe_key", side_effect=[(429, "temporary", {}), (200, "", {})]), \
                 patch.object(check_keys.time, "sleep"), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(check_keys.main(), 0)
            self.assertIn("summary: 1/1 alive", output.getvalue())

    def test_known_statuses(self):
        self.assertEqual(classify_status(200, ""), "OK")
        self.assertIn("invalid", classify_status(401, ""))
        self.assertIn("balance exhausted", classify_status(402, ""))
        self.assertIn("forbidden", classify_status(403, ""))
        self.assertIn("rate limited", classify_status(429, ""))
        self.assertIn("network/timeout", classify_status(None, ""))

    def test_5xx_is_fake_dead(self):
        self.assertIn("server error", classify_status(503, ""))

    def test_unexpected_status_reports_code(self):
        self.assertIn("HTTP 418", classify_status(418, "teapot"))


if __name__ == "__main__":
    unittest.main()

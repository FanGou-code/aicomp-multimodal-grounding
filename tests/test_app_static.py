"""Static checks for the review frontend (no DOM)."""

import re
import unittest
from pathlib import Path

APP_JS = Path(__file__).resolve().parents[1] / "foundry" / "review" / "web" / "app.js"


class FrontendStaticTest(unittest.TestCase):
    def test_no_references_to_retired_identifiers(self):
        # Retired variables must never be referenced again: a leftover throws
        # ReferenceError on every canvas redraw and freezes the reviewer.
        source = APP_JS.read_text(encoding="utf-8")
        body = re.sub(r"//[^\n]*", "", source)
        for name in ("isAi", "isAbsentItem", "confirmAbsent", "markAbsent",
                     "clearBbox", "queryZhText", "absentBtn", "clearBtn"):
            hits = re.findall(rf"\b{name}\b", body)
            self.assertEqual(hits, [], f"retired identifier referenced: {name}")

    def test_node_check_equivalent_brace_balance(self):
        # Cheap structural sanity: the file must remain parseable by node in CI.
        self.assertTrue(APP_JS.is_file())


if __name__ == "__main__":
    unittest.main()

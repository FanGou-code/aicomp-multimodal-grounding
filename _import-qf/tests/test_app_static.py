"""Static checks for the review frontend (no DOM)."""

import re
import shutil
import subprocess
import unittest
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1] / "foundry" / "review" / "web"
APP_JS = WEB_DIR / "app.js"
INDEX_HTML = WEB_DIR / "index.html"


class FrontendStaticTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "frontend state tests require Node.js")
    def test_real_frontend_save_state_transitions(self):
        result = subprocess.run(
            [shutil.which("node"), str(Path(__file__).with_name("review_frontend.cjs"))],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_references_to_retired_identifiers(self):
        # Retired variables must never be referenced again: a leftover throws
        # ReferenceError/TypeError on startup or canvas redraw and freezes the reviewer.
        source = APP_JS.read_text(encoding="utf-8")
        body = re.sub(r"//[^\n]*", "", source)
        for name in (
            "isAi", "isAbsentItem", "confirmAbsent", "markAbsent",
            "clearBbox", "queryZhText", "absentBtn", "clearBtn",
            "corpusSwitch", "setCorpusFilter", "STORAGE_KEY_CORPUS",
        ):
            hits = re.findall(rf"\b{name}\b", body)
            self.assertEqual(hits, [], f"retired identifier referenced: {name}")

    def test_dom_references_consistency(self):
        # Every dom.<property> access in app.js must be declared in the dom map,
        # and every declared element id must exist in index.html.
        app_source = APP_JS.read_text(encoding="utf-8")
        html_source = INDEX_HTML.read_text(encoding="utf-8")

        dom_block_match = re.search(r"const dom = \{([\s\S]*?)\n  \};", app_source)
        self.assertIsNotNone(dom_block_match, "const dom = {...} not found in app.js")
        dom_block = dom_block_match.group(1)

        declared_elements = dict(
            re.findall(r"([a-zA-Z0-9_$]+)\s*:\s*document\.getElementById\([\"'](.*?)[\"']\)", dom_block)
        )
        # All dom.<property> accesses must be defined
        used_props = set(re.findall(r"dom\.([a-zA-Z0-9_$]+)", app_source))
        undefined = used_props - set(declared_elements) - {"toastContainer"}
        self.assertEqual(undefined, set(), f"dom properties accessed but not defined in const dom: {undefined}")

        # All declared element IDs must be present in index.html
        for prop, elem_id in declared_elements.items():
            self.assertTrue(
                f'id="{elem_id}"' in html_source or f"id='{elem_id}'" in html_source,
                f"Element id='{elem_id}' (dom.{prop}) declared in app.js but not found in index.html",
            )

    def test_node_check_syntax(self):
        # If node is available on PATH, verify app.js syntax
        node_bin = shutil.which("node")
        if node_bin:
            res = subprocess.run([node_bin, "-c", str(APP_JS)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"app.js failed node syntax check:\n{res.stderr}")


if __name__ == "__main__":
    unittest.main()

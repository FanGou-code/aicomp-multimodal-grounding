"""Contract tests for the ordinal prompt files and their decoders."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aicomp_grounding.ordinal import enumerate as ordinal_enumerate
from aicomp_grounding.ordinal import loader, parse

HEX64 = re.compile(r"^[0-9a-f]{64}$")


class LoaderTests(unittest.TestCase):
    def test_every_prompt_splits_into_system_and_user(self):
        for name in loader.PROMPT_NAMES:
            with self.subTest(name=name):
                system, user = loader.load_prompt(name)
                self.assertTrue(system.strip())
                self.assertTrue(user.strip())

    def test_parse_prompt_takes_a_query_placeholder(self):
        _, user = loader.load_prompt(loader.PARSE)
        self.assertIn("{query}", user)
        self.assertNotIn("{category}", user)

    def test_enumerate_prompt_takes_a_category_placeholder(self):
        _, user = loader.load_prompt(loader.ENUMERATE)
        self.assertIn("{category}", user)
        self.assertNotIn("{query}", user)

    def test_hashes_cover_every_prompt_and_are_stable(self):
        hashes = loader.prompt_hashes()
        self.assertEqual(set(hashes), set(loader.PROMPT_NAMES))
        for value in hashes.values():
            self.assertRegex(value, HEX64)
        self.assertEqual(hashes, loader.prompt_hashes())
        self.assertRegex(loader.prompts_fingerprint(), HEX64)

    def test_unknown_prompt_name_is_rejected(self):
        with self.assertRaises(ValueError):
            loader.load_prompt("nope")

    def test_missing_or_malformed_file_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(loader, "PROMPT_DIR", Path(tmp)):
                with self.assertRaises(FileNotFoundError):
                    loader.load_prompt(loader.PARSE)
                (Path(tmp) / "parse.md").write_text("no markers here", encoding="utf-8")
                with self.assertRaises(ValueError):
                    loader.load_prompt(loader.PARSE)
                (Path(tmp) / "parse.md").write_text("[system]\ns\n[user]\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    loader.load_prompt(loader.PARSE)


class ParseMessageTests(unittest.TestCase):
    def test_parse_messages_are_text_only_and_carry_the_query(self):
        messages = parse.build_parse_messages("the second car from the left")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("the second car from the left", messages[1]["content"][0]["text"])
        self.assertNotIn("{query}", messages[1]["content"][0]["text"])
        for message in messages:
            for part in message["content"]:
                self.assertEqual(part["type"], "text")

    def test_strict_json_object_accepts_one_object_only(self):
        self.assertEqual(parse.strict_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(parse.strict_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        for bad in (None, 5, "[]", "[1, 2]", "not json", '{"a": 1} trailing',
                    '{"a": 1, "a": 2}'):
            with self.subTest(bad=bad):
                self.assertIsNone(parse.strict_json_object(bad))


class EnumerateMessageTests(unittest.TestCase):
    def test_enumerate_messages_carry_one_image_and_the_category(self):
        marker = object()
        messages = ordinal_enumerate.build_enumerate_messages(marker, "person")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        parts = messages[1]["content"]
        self.assertEqual(parts[0], {"type": "image", "image": marker})
        self.assertIn("person", parts[1]["text"])
        self.assertNotIn("{category}", parts[1]["text"])

    def test_decode_run_requires_the_exact_envelope(self):
        good = '{"count": 2, "instances": [{"bbox": [0.1,0.1,0.2,0.2], "confidence": 0.5}]}'
        self.assertEqual(ordinal_enumerate.decode_run(good)["count"], 2)
        bad = [
            '{"instances": []}',
            '{"count": 0, "instances": [], "extra": 1}',
            '{"count": "2", "instances": []}',
            '{"count": true, "instances": []}',
            '{"count": 0, "instances": {}}',
            "not json",
        ]
        for payload in bad:
            with self.subTest(payload=payload):
                self.assertIsNone(ordinal_enumerate.decode_run(payload))


if __name__ == "__main__":
    unittest.main()

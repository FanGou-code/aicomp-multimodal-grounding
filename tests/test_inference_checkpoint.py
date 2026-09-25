"""Resume-default tests for the inference entrypoint."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class InferenceResumeDefaultTests(unittest.TestCase):
    def test_offline_infer_resume_defaults_to_true(self):
        from tools.infer import parse_args

        with mock.patch("sys.argv", ["infer.py"]):
            args = parse_args()
            self.assertTrue(args.resume)

        with mock.patch("sys.argv", ["infer.py", "--no-resume"]):
            args = parse_args()
            self.assertFalse(args.resume)

        with mock.patch("sys.argv", ["infer.py", "--resume"]):
            args = parse_args()
            self.assertTrue(args.resume)


if __name__ == "__main__":
    unittest.main()

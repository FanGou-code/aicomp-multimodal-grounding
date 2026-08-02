import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.leaderboard import (
    format_leaderboard_table,
    load_leaderboard_registry,
    register_score,
)


class TestLeaderboardRegistry(unittest.TestCase):
    def test_register_and_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "leaderboard_registry.json"
            data = load_leaderboard_registry(reg_path)
            self.assertEqual(data["submissions"], [])

            record = register_score(
                official_score=0.6471,
                tag="Base",
                notes="Initial test",
                inference_run_id="infer_base_test",
                registry_path=reg_path,
            )
            self.assertEqual(record["official_score"], 0.6471)
            self.assertEqual(record["tag"], "Base")

            table = format_leaderboard_table(reg_path)
            self.assertIn("0.6471", table)
            self.assertIn("infer_base_test", table)


if __name__ == "__main__":
    unittest.main()

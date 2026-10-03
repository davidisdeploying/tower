"""Static regression tests for current Tower MCP governance descriptions."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


class GovernanceDescriptionTests(unittest.TestCase):
    def test_server_descriptions_do_not_instruct_retired_vault_workflow(self):
        text = (ROOT / "server.py").read_text(encoding="utf-8")
        for retired in (
            "get_note('homelab-vault/NOW.md')",
            "loupe-vault/DECISIONS.md",
            "library/to-<seat>",
            "Library relay prompt",
        ):
            self.assertNotIn(retired, text)

    def test_live_relay_comments_name_homelab_vault(self):
        text = (ROOT / "vaultsearch.py").read_text(encoding="utf-8")
        self.assertNotIn("into library's to-*/from-* relay lanes", text)
        self.assertIn("homelab-vault/to-<seat>/<lane>/latest.md", text)


if __name__ == "__main__":
    unittest.main()

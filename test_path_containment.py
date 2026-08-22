import os
import tempfile
import unittest
from unittest import mock

import vaultsearch as vs


class PathContainmentTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.outside = tempfile.TemporaryDirectory()
        self.addCleanup(self.outside.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.patcher = mock.patch.object(vs, "VAULTS_ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def _escape_link(self, relative):
        link = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(self.outside.name, link)
        return link

    def test_shared_resolver_accepts_inside_and_rejects_symlink_escape(self):
        os.makedirs(os.path.join(self.root, "safe"))
        self.assertEqual(vs._resolve("safe/note.md"), os.path.join(self.root, "safe", "note.md"))
        self._escape_link("escape")
        with self.assertRaisesRegex(ValueError, "escapes vaults root"):
            vs._resolve("escape/secret.md")

    def test_write_and_delete_reject_real_symlink_escape(self):
        self._escape_link("demo-vault/outside")
        written = vs.write_note("demo-vault/outside/secret.md", "secret")
        deleted = vs.delete_note("demo-vault/outside/secret.md")
        self.assertFalse(written["ok"])
        self.assertFalse(deleted["ok"])
        self.assertIn("escapes vaults root", written["error"])
        self.assertIn("escapes vaults root", deleted["error"])
        self.assertFalse(os.path.exists(os.path.join(self.outside.name, "secret.md")))

    def test_stage_prompt_rejects_relay_directory_symlink_escape(self):
        os.makedirs(os.path.join(self.root, "homelab-vault"))
        self._escape_link("homelab-vault/to-worker1")
        token = "FLEET-WORKER1-BUILD-20260722-containment"
        result = vs.stage_prompt("worker1", "prompts", token, token + "\nbody\n")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "path escapes vaults root")
        self.assertFalse(os.path.exists(os.path.join(self.outside.name, "prompts", "latest.md")))

    def test_journal_rejects_vault_symlink_escape(self):
        self._escape_link("demo-vault")
        result = vs.append_journal_entry(
            "demo-vault", "decisions", "codex",
            "2026-07-22 18:11 UTC / 13:11 CDT", "must stay contained",
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "vault path escapes vaults root")
        self.assertFalse(os.path.exists(os.path.join(self.outside.name, "journal")))


if __name__ == "__main__":
    unittest.main()

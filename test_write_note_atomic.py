import os
import stat
import tempfile
import unittest
from unittest import mock

import vaultsearch as vs


class AtomicWriteNoteTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.original_root = vs.VAULTS_ROOT
        vs.VAULTS_ROOT = os.path.realpath(self.tempdir.name)
        self.addCleanup(setattr, vs, "VAULTS_ROOT", self.original_root)

    def _full(self, path):
        return os.path.join(vs.VAULTS_ROOT, path)

    def test_overwrite_replaces_content_and_preserves_mode(self):
        path = "homelab-vault/sessions/example.md"
        full = self._full(path)
        os.makedirs(os.path.dirname(full))
        with open(full, "w", encoding="utf-8") as handle:
            handle.write("old\n")
        os.chmod(full, 0o640)

        with mock.patch.object(vs.os, "fsync", wraps=os.fsync) as fsync:
            result = vs.write_note(path, "new\n", mode="overwrite")

        self.assertTrue(result["ok"], result)
        with open(full, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "new\n")
        self.assertEqual(stat.S_IMODE(os.stat(full).st_mode), 0o640)
        self.assertGreaterEqual(fsync.call_count, 2)
        self.assertEqual(os.listdir(os.path.dirname(full)), ["example.md"])

    def test_new_file_uses_vault_writable_mode(self):
        path = "homelab-vault/audits/new.md"
        result = vs.write_note(path, "created\n", mode="overwrite")

        self.assertTrue(result["ok"], result)
        self.assertEqual(stat.S_IMODE(os.stat(self._full(path)).st_mode), 0o664)

    def test_replace_failure_preserves_original_and_cleans_temp(self):
        path = "homelab-vault/sessions/protected.md"
        full = self._full(path)
        parent = os.path.dirname(full)
        os.makedirs(parent)
        with open(full, "w", encoding="utf-8") as handle:
            handle.write("original\n")

        with mock.patch.object(vs.os, "replace", side_effect=OSError("injected replace failure")):
            result = vs.write_note(path, "replacement\n", mode="overwrite")

        self.assertFalse(result["ok"])
        self.assertIn("injected replace failure", result["error"])
        with open(full, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "original\n")
        self.assertEqual(os.listdir(parent), ["protected.md"])

    def test_append_and_prepend_behavior_is_unchanged(self):
        path = "homelab-vault/sessions/modes.md"
        self.assertTrue(vs.write_note(path, "middle\n", mode="overwrite")["ok"])
        self.assertTrue(vs.write_note(path, "tail\n", mode="append")["ok"])
        self.assertTrue(vs.write_note(path, "head\n", mode="prepend")["ok"])

        with open(self._full(path), "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "head\nmiddle\ntail\n")


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from unittest import mock

import server
import vaultsearch as vs


class RangedReadTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.patcher = mock.patch.object(vs, "VAULTS_ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        os.makedirs(os.path.join(self.root, "homelab-vault", "notes"))
        os.makedirs(os.path.join(self.root, "homelab-vault", "from-worker2", "transcripts"))
        self.note_rel = "homelab-vault/notes/sample.md"
        self.note = os.path.join(self.root, self.note_rel)
        with open(self.note, "w", encoding="utf-8") as handle:
            handle.write("alpha\nbeta\ngamma\ndelta\n")
        self.json_rel = "homelab-vault/from-worker2/transcripts/run.json"
        self.json_path = os.path.join(self.root, self.json_rel)
        with open(self.json_path, "w", encoding="utf-8") as handle:
            handle.write('{"n":1}\n{"n":2}\n')

    def test_byte_pages_are_exact_and_drift_metadata_is_stable(self):
        first = vs.read_vault_range(self.note_rel, offset=0, limit=6, unit="bytes")
        self.assertTrue(first["ok"], first)
        self.assertEqual(first["content"], "alpha\n")
        self.assertEqual(first["returned_range"], [0, 6])
        self.assertEqual(first["next_offset"], 6)
        self.assertFalse(first["eof"])
        second = vs.read_vault_range(
            self.note_rel, offset=first["next_offset"], limit=100, unit="bytes",
            expected_sha256=first["sha256"], expected_mtime_ns=first["mtime_ns"],
        )
        self.assertTrue(second["ok"], second)
        self.assertEqual(second["content"], "beta\ngamma\ndelta\n")
        self.assertTrue(second["eof"])
        self.assertEqual(second["sha256"], first["sha256"])

    def test_line_pages_report_total_and_complete_ranges(self):
        page = vs.read_vault_range(self.note_rel, offset=1, limit=2, unit="lines", max_bytes=64)
        self.assertTrue(page["ok"], page)
        self.assertEqual(page["content"], "beta\ngamma\n")
        self.assertEqual(page["total_lines"], 4)
        self.assertEqual(page["returned_range"], [1, 3])
        self.assertEqual(page["next_offset"], 3)
        self.assertFalse(page["eof"])

    def test_mid_pagination_drift_fails_closed_without_content(self):
        first = vs.read_vault_range(self.note_rel, limit=5)
        with open(self.note, "a", encoding="utf-8") as handle:
            handle.write("changed\n")
        second = vs.read_vault_range(
            self.note_rel, offset=first["next_offset"], limit=5,
            expected_sha256=first["sha256"], expected_mtime_ns=first["mtime_ns"],
        )
        self.assertFalse(second["ok"])
        self.assertEqual(second["error"], "content changed")
        self.assertNotIn("content", second)
        self.assertNotEqual(second["sha256"], first["sha256"])

    def test_json_transcript_supported_and_payload_caps_clamp(self):
        page = vs.read_vault_range(self.json_rel, limit=999999, max_bytes=7)
        self.assertTrue(page["ok"], page)
        self.assertEqual(page["returned_bytes"], 7)
        self.assertEqual(page["max_bytes"], 7)
        self.assertEqual(page["next_offset"], 7)

    def test_oversize_line_does_not_break_line_cursor(self):
        with open(self.json_path, "w", encoding="utf-8") as handle:
            handle.write("x" * 200 + "\nsmall\n")
        page = vs.read_vault_range(self.json_rel, unit="lines", limit=2, max_bytes=32)
        self.assertTrue(page["ok"], page)
        self.assertEqual(page["returned_units"], 0)
        self.assertEqual(page["next_offset"], 0)
        self.assertTrue(page["blocked_by_oversize_line"])
        self.assertFalse(page["eof"])

    def test_path_and_argument_validation_and_symlink_escape(self):
        bad = (
            vs.read_vault_range("/etc/passwd.md"),
            vs.read_vault_range("../escape.md"),
            vs.read_vault_range("homelab-vault/notes/no.txt"),
            vs.read_vault_range(self.note_rel, unit="words"),
            vs.read_vault_range(self.note_rel, offset=-1),
            vs.read_vault_range(self.note_rel, expected_sha256="BAD"),
        )
        self.assertTrue(all(not result["ok"] for result in bad))
        outside = os.path.join(self.tmp.name + "-outside.md")
        with open(outside, "w", encoding="utf-8") as handle:
            handle.write("secret")
        self.addCleanup(lambda: os.path.exists(outside) and os.unlink(outside))
        link = os.path.join(self.root, "homelab-vault", "notes", "escape.md")
        os.symlink(outside, link)
        escaped = vs.read_vault_range("homelab-vault/notes/escape.md")
        self.assertFalse(escaped["ok"])
        self.assertIn("escapes vaults root", escaped["error"])

    def test_utf8_boundary_replacement_is_reported(self):
        with open(self.note, "w", encoding="utf-8") as handle:
            handle.write("AéB")
        page = vs.read_vault_range(self.note_rel, offset=2, limit=1)
        self.assertTrue(page["ok"], page)
        self.assertTrue(page["decode_replacements"])

    def test_server_wrapper_preserves_arguments(self):
        with mock.patch.object(vs, "read_vault_range", return_value={"ok": True}) as read:
            result = server.read_vault_range(self.note_rel, 1, 2, "lines", 100, "a" * 64, 3)
        self.assertTrue(result["ok"])
        read.assert_called_once_with(self.note_rel, 1, 2, "lines", 100, "a" * 64, 3)


if __name__ == "__main__":
    unittest.main()

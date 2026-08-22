import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import server
import vaultsearch as vs


STAMP = "2026-07-22 17:24 UTC / 12:24 CDT"


class JournalAppendTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.original_root = vs.VAULTS_ROOT
        vs.VAULTS_ROOT = os.path.realpath(self.tempdir.name)
        self.addCleanup(setattr, vs, "VAULTS_ROOT", self.original_root)
        for vault in ("homelab-vault", "loupe-vault"):
            os.makedirs(os.path.join(vs.VAULTS_ROOT, vault))

    def _append(self, text="decision recorded", **overrides):
        args = {
            "vault": "homelab-vault",
            "kind": "decisions",
            "seat": "codex",
            "stamp": STAMP,
            "text": text,
        }
        args.update(overrides)
        return vs.append_journal_entry(**args)

    def _full(self, result):
        return os.path.join(vs.VAULTS_ROOT, result["path"])

    def test_valid_entry_uses_derived_path_and_exact_format(self):
        result = self._append("Tower journal append is safe")
        self.assertTrue(result["ok"], result)
        self.assertEqual(
            result["path"],
            "homelab-vault/journal/inbox/decisions/codex-2026-07-22.md",
        )
        with open(self._full(result), "r", encoding="utf-8") as handle:
            self.assertEqual(
                handle.read(),
                "- 2026-07-22 17:24 UTC / 12:24 CDT — Tower journal append is safe\n",
            )

    def test_second_entry_appends_without_overwrite(self):
        first = self._append("first")
        second = self._append("second")
        self.assertTrue(first["ok"] and second["ok"])
        with open(self._full(second), "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].endswith("— first\n"))
        self.assertTrue(lines[1].endswith("— second\n"))

    def test_existing_missing_newline_is_repaired(self):
        path = os.path.join(
            vs.VAULTS_ROOT,
            "homelab-vault/journal/inbox/decisions/codex-2026-07-22.md",
        )
        os.makedirs(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("legacy-without-newline")
        result = self._append("new entry")
        self.assertTrue(result["ok"], result)
        with open(path, "r", encoding="utf-8") as handle:
            self.assertEqual(
                handle.read(),
                "legacy-without-newline\n- 2026-07-22 17:24 UTC / 12:24 CDT — new entry\n",
            )

    def test_raw_write_note_is_blocked_for_all_inbox_modes(self):
        path = "homelab-vault/journal/inbox/decisions/codex-2026-07-22.md"
        for mode in ("overwrite", "append", "prepend"):
            with self.subTest(mode=mode):
                result = vs.write_note(path, "unsafe\n", mode=mode)
                self.assertFalse(result["ok"])
                self.assertIn("append_journal_entry", result["error"])
        self.assertFalse(os.path.exists(os.path.join(vs.VAULTS_ROOT, path)))

    def test_invalid_inputs_are_rejected(self):
        cases = (
            {"vault": "../homelab-vault"},
            {"vault": "missing-vault"},
            {"kind": "monthly"},
            {"seat": "worker5"},
            {"stamp": "2026-07-22 17:24 UTC / 11:24 CDT"},
            {"stamp": "not-a-stamp"},
            {"text": "two\nlines"},
            {"text": ""},
            {"text": "x" * 5000},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                result = self._append(**overrides)
                self.assertFalse(result["ok"], result)

    def test_concurrent_entries_are_complete_and_unique(self):
        texts = [f"entry-{index:02d}" for index in range(24)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(self._append, texts))
        self.assertTrue(all(result["ok"] for result in results), results)
        with open(self._full(results[0]), "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        self.assertEqual(len(lines), len(texts))
        self.assertEqual(
            {line.rsplit("— ", 1)[1].strip() for line in lines},
            set(texts),
        )

    def test_mcp_tool_wrapper_uses_same_validation(self):
        result = server.append_journal_entry(
            "loupe-vault",
            "learnings",
            "codex",
            STAMP,
            "wrapper verified",
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(
            result["path"],
            "loupe-vault/journal/inbox/learnings/codex-2026-07-22.md",
        )


if __name__ == "__main__":
    unittest.main()

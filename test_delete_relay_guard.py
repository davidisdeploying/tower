"""Focused tests for vaultsearch.delete_note's relay-machinery protection
(FLEET-WORKER2-BUILD-20260710-delete-relay-guard).

Covers: all five current seats under Library to-*/from-*, the
preserved legacy loupe-vault to-*/from-* lanes, nested files inside those
lanes, ordinary note deletion (still soft-deletes), traversal/absolute
path rejection, and that a rejected delete never mutates the vault.

Uses a tempdir monkeypatched onto vaultsearch.VAULTS_ROOT — no live vault,
no network, no NAS. Run with:
    python -m unittest test_delete_relay_guard -v
"""
import os
import shutil
import tempfile
import unittest

import vaultsearch as vs

SEATS = ("delta", "charlie", "localworker", "worker4", "alpha")


class DeleteRelayGuardTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fleet-delete-relay-guard-test-")
        self._orig_root = vs.VAULTS_ROOT
        vs.VAULTS_ROOT = os.path.realpath(self.tmp)

    def tearDown(self):
        vs.VAULTS_ROOT = self._orig_root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _full(self, rel_path: str) -> str:
        return os.path.join(vs.VAULTS_ROOT, rel_path)

    def _touch(self, rel_path: str, content: str = "x\n") -> str:
        full = self._full(rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return full

    def _assert_protected(self, rel_path: str):
        full = self._touch(rel_path)
        result = vs.delete_note(rel_path)
        self.assertFalse(result["ok"], (rel_path, result))
        self.assertEqual(result["error"], "protected path")
        self.assertTrue(os.path.isfile(full), f"file moved despite rejection: {rel_path}")
        # no .trash dir should exist at all — nothing was moved anywhere.
        self.assertFalse(os.path.isdir(self._full(".trash")))

    # --- current relay: homelab-vault/to-<seat>|from-<seat> ------------

    def test_charlie_compendium_all_seats_both_directions_protected(self):
        for seat in SEATS:
            for direction in ("to", "from"):
                path = f"homelab-vault/{direction}-{seat}/prompts/latest.md"
                self._assert_protected(path)

    def test_charlie_compendium_recon_lane_protected(self):
        for seat in SEATS:
            self._assert_protected(f"homelab-vault/to-{seat}/recon/latest.md")
            self._assert_protected(f"homelab-vault/from-{seat}/recon/report.md")

    def test_charlie_compendium_nested_files_protected(self):
        for seat in SEATS:
            self._assert_protected(
                f"homelab-vault/from-{seat}/prompts/archive/2026-07-10/old-response.md"
            )
            self._assert_protected(
                f"homelab-vault/to-{seat}/runs/deeply/nested/artifact.md"
            )

    # --- preserved legacy relay: loupe-vault/to-<seat>|from-<seat> ---------

    def test_loupe_vault_legacy_seats_protected(self):
        # worker1/worker3 were the historical CC-era seats but the guard must not be
        # a finite list of just those two — it must cover the full current
        # roster under the legacy vault too.
        for seat in SEATS:
            for direction in ("to", "from"):
                path = f"loupe-vault/{direction}-{seat}/session.md"
                self._assert_protected(path)

    def test_loupe_vault_nested_legacy_files_protected(self):
        self._assert_protected("loupe-vault/from-worker1/runs/2026-06-01/artifact.md")
        self._assert_protected("loupe-vault/to-worker3/archive/old/prompt.md")

    # --- existing protections preserved -------------------------------------

    def test_trash_itself_protected(self):
        path = ".trash/20260101T000000Z/note.md"
        full = self._touch(path)
        result = vs.delete_note(path)
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["error"], "protected path")
        self.assertTrue(os.path.isfile(full))

    def test_protected_basenames_preserved(self):
        for name in (
            "DECISIONS.md",
            "LEARNINGS.md",
            "prompts.md",
            "responses.md",
            "latest.md",
            "latest_response.md",
            "recon.md",
        ):
            self._assert_protected(f"loupe-vault/{name}")

    # --- ordinary notes still soft-delete -----------------------------------

    def test_ordinary_note_outside_relay_tree_soft_deletes(self):
        full = self._touch("loupe-vault/notes/random-note.md", "hello\n")
        result = vs.delete_note("loupe-vault/notes/random-note.md")
        self.assertTrue(result["ok"], result)
        self.assertFalse(os.path.isfile(full))
        trashed = self._full(result["trashed_to"])
        self.assertTrue(os.path.isfile(trashed))
        with open(trashed, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "hello\n")

    def test_ordinary_session_note_soft_deletes(self):
        self._touch("homelab-vault/sessions/2026-07-10-some-arc.md", "note\n")
        result = vs.delete_note("homelab-vault/sessions/2026-07-10-some-arc.md")
        self.assertTrue(result["ok"], result)

    def test_ordinary_audits_note_soft_deletes(self):
        self._touch("homelab-vault/audits/2026-07-10-check.md", "audit\n")
        result = vs.delete_note("homelab-vault/audits/2026-07-10-check.md")
        self.assertTrue(result["ok"], result)

    def test_sibling_vault_unaffected(self):
        self._touch("homelab-vault/notes/foo.md", "foo\n")
        result = vs.delete_note("homelab-vault/notes/foo.md")
        self.assertTrue(result["ok"], result)

    def test_seat_named_dir_outside_relay_vault_not_protected(self):
        # A directory literally named "to-worker2" is only relay machinery when
        # it lives under a relay vault root — elsewhere it's an ordinary dir.
        self._touch("prospect-vault/to-worker2/notes.md", "x\n")
        result = vs.delete_note("prospect-vault/to-worker2/notes.md")
        self.assertTrue(result["ok"], result)

    # --- traversal / absolute path protections ------------------------------

    def test_absolute_path_rejected(self):
        result = vs.delete_note("/etc/passwd")
        self.assertFalse(result["ok"])
        self.assertIn("absolute path rejected", result["error"])

    def test_dotdot_component_rejected(self):
        result = vs.delete_note("loupe-vault/../../../etc/passwd")
        self.assertFalse(result["ok"])
        self.assertIn("'..' component rejected", result["error"])

    def test_dotdot_targeting_relay_tree_rejected(self):
        result = vs.delete_note("homelab-vault/../homelab-vault/to-worker2/prompts/latest.md")
        self.assertFalse(result["ok"])
        self.assertIn("'..' component rejected", result["error"])

    def test_non_md_rejected(self):
        self._touch("loupe-vault/notes/file.txt")
        # rename doesn't matter — delete_note requires .md suffix regardless.
        result = vs.delete_note("loupe-vault/notes/file.txt")
        self.assertFalse(result["ok"])
        self.assertIn("not a .md file", result["error"])

    # --- no relay mutation on any rejected attempt ---------------------------

    def test_no_relay_mutation_across_full_matrix(self):
        # Populate every seat/direction/vault combination, attempt delete on
        # all of them, then assert the entire relay tree is byte-identical.
        paths = []
        for vault in ("homelab-vault", "loupe-vault"):
            for seat in SEATS:
                for direction in ("to", "from"):
                    p = f"{vault}/{direction}-{seat}/prompts/latest.md"
                    paths.append(p)
                    self._touch(p, f"content for {p}\n")

        before = {}
        for root, _dirs, files in os.walk(vs.VAULTS_ROOT):
            for name in files:
                full = os.path.join(root, name)
                with open(full, "r", encoding="utf-8") as f:
                    before[full] = f.read()

        for p in paths:
            result = vs.delete_note(p)
            self.assertFalse(result["ok"], p)

        after = {}
        for root, _dirs, files in os.walk(vs.VAULTS_ROOT):
            for name in files:
                full = os.path.join(root, name)
                with open(full, "r", encoding="utf-8") as f:
                    after[full] = f.read()

        self.assertEqual(before, after)
        self.assertFalse(os.path.isdir(self._full(".trash")))


if __name__ == "__main__":
    unittest.main()

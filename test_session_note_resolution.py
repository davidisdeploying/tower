"""H7 regression: session-note paths into a sibling vault must resolve there.

The defect was reported as a "library/..." prefix problem and so looked
vault-specific. It was not. _resolve_session_note only ever stripped the relay
root's OWN directory name, so a path into any other vault fell through to
os.path.join(VAULT, sn) and landed at
VAULT/<other>-vault/sessions/... — a stray nested stub under the relay root, with
the worker's append never reaching the real note. library is retired,
which is why the original symptom stopped being visible; the defect did not stop
happening, it just moved to whichever vault was in use. Prospect
(prospect-vault/sessions/...) is the one it was actually losing.

Read-only: every case below is pure path arithmetic against real directories
under ~/Vaults. Run with:
    python -m unittest test_session_note_resolution -v
"""
import os
import tempfile
import unittest
from unittest import mock

import kicker


class SessionNoteResolutionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.relay = os.path.join(self.root, "homelab-vault")
        os.makedirs(os.path.join(self.relay, "sessions"))
        os.makedirs(os.path.join(self.root, "prospect-vault", "sessions"))
        patcher = mock.patch.object(kicker, "VAULT", self.relay)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_project_relative_path_stays_in_the_relay_root(self):
        self.assertEqual(
            kicker._resolve_session_note("sessions/note.md"),
            os.path.join(self.relay, "sessions", "note.md"),
        )

    def test_relay_root_prefix_is_stripped_not_doubled(self):
        self.assertEqual(
            kicker._resolve_session_note("homelab-vault/sessions/note.md"),
            os.path.join(self.relay, "sessions", "note.md"),
        )

    def test_bare_slug_becomes_a_relay_root_session(self):
        self.assertEqual(
            kicker._resolve_session_note("note"),
            os.path.join(self.relay, "sessions", "note.md"),
        )

    def test_sibling_vault_resolves_to_that_vault(self):
        """The H7 regression itself."""
        resolved = kicker._resolve_session_note("prospect-vault/sessions/note.md")
        self.assertEqual(
            resolved, os.path.join(self.root, "prospect-vault", "sessions", "note.md")
        )
        self.assertNotIn(
            os.path.join("homelab-vault", "prospect-vault"), resolved,
            "a sibling vault must not be nested under the relay root",
        )

    def test_unknown_vault_name_does_not_invent_a_directory(self):
        """Gated on the directory existing, so a typo keeps the old behaviour
        rather than silently resolving somewhere new."""
        self.assertEqual(
            kicker._resolve_session_note("nonexistent-vault/sessions/note.md"),
            os.path.join(self.relay, "nonexistent-vault", "sessions", "note.md"),
        )

    def test_ordinary_subdirectory_is_not_mistaken_for_a_vault(self):
        os.makedirs(os.path.join(self.relay, "audits"))
        self.assertEqual(
            kicker._resolve_session_note("audits/2026-08-09.md"),
            os.path.join(self.relay, "audits", "2026-08-09.md"),
        )

    def test_absolute_paths_pass_through(self):
        absolute = os.path.join(self.root, "prospect-vault", "sessions", "x.md")
        self.assertEqual(kicker._resolve_session_note(absolute), absolute)

    def test_empty_input_resolves_to_empty(self):
        self.assertEqual(kicker._resolve_session_note("   "), "")

    def test_symlinked_vault_escaping_the_root_is_refused(self):
        """Containment: widening resolution by one level must not become a way
        out of ~/Vaults."""
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        os.symlink(outside.name, os.path.join(self.root, "escape-vault"))
        resolved = kicker._resolve_session_note("escape-vault/sessions/note.md")
        self.assertEqual(
            resolved, os.path.join(self.relay, "escape-vault", "sessions", "note.md"),
            "an escaping symlink falls back to the relay root, never outside the vaults root",
        )
        self.assertFalse(os.path.realpath(resolved).startswith(os.path.realpath(outside.name)))


if __name__ == "__main__":
    unittest.main()

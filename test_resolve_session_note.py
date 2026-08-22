"""Focused tests for kicker._resolve_session_note
(FLEET-WORKER2-BUILD-20260721-to3c-fix-session-note-normalization).

Proven defect: a Vaults-relative reference prefixed with the project dir name
(e.g. "homelab-vault/sessions/x.md") was joined onto VAULT unstripped,
doubling the project dir (VAULT/homelab-vault/sessions/x.md). This suite
pins the fixed normalization: both the project-relative spelling
("sessions/x.md") and the Vaults-relative spelling ("homelab-vault/sessions/
x.md") must resolve to the identical VAULT/sessions/x.md path.

Uses a tempdir monkeypatched onto kicker.VAULT — no live vault, no network,
no workers launched. Run with:
    python -m unittest test_resolve_session_note -v
"""
import os
import tempfile
import unittest

import kicker


class ResolveSessionNoteTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fleet-resolve-session-note-test-")
        # Mirror the real deployment: VAULT is a directory named
        # "homelab-vault" (the project-dir-prefix check keys off basename).
        self.vault = os.path.join(self.tmp, "homelab-vault")
        os.makedirs(self.vault, exist_ok=True)
        self._orig_vault = kicker.VAULT
        kicker.VAULT = self.vault

    def tearDown(self):
        kicker.VAULT = self._orig_vault

    # --- bare slug -----------------------------------------------------

    def test_bare_slug_no_extension(self):
        result = kicker._resolve_session_note("2026-07-21-slug")
        self.assertEqual(
            result, os.path.join(self.vault, "sessions", "2026-07-21-slug.md")
        )

    def test_bare_slug_md_suffix_not_doubled(self):
        result = kicker._resolve_session_note("2026-07-21-slug.md")
        self.assertEqual(
            result, os.path.join(self.vault, "sessions", "2026-07-21-slug.md")
        )

    # --- project-relative ------------------------------------------------

    def test_project_relative_path(self):
        result = kicker._resolve_session_note("sessions/2026-07-21-slug.md")
        self.assertEqual(
            result, os.path.join(self.vault, "sessions", "2026-07-21-slug.md")
        )

    # --- Vaults-relative, project-dir-prefixed ----------------------------

    def test_vault_prefixed_path_no_doubling(self):
        result = kicker._resolve_session_note(
            "homelab-vault/sessions/2026-07-21-slug.md"
        )
        expected = os.path.join(self.vault, "sessions", "2026-07-21-slug.md")
        self.assertEqual(result, expected)
        # The defect this guards against: VAULT/homelab-vault/sessions/...
        self.assertNotIn(
            os.path.join("homelab-vault", "homelab-vault"), result
        )

    def test_project_relative_and_vault_prefixed_are_equal(self):
        project_relative = kicker._resolve_session_note(
            "sessions/2026-07-21-slug.md"
        )
        vault_prefixed = kicker._resolve_session_note(
            "homelab-vault/sessions/2026-07-21-slug.md"
        )
        self.assertEqual(project_relative, vault_prefixed)

    def test_prefix_must_match_exactly_not_substring(self):
        # A dir name that merely starts with "homelab-vault" (no separator)
        # must NOT have the prefix stripped — only an exact "homelab-vault/"
        # segment prefix qualifies.
        result = kicker._resolve_session_note(
            "homelab-vault-archive/sessions/2026-07-21-slug.md"
        )
        expected = os.path.join(
            self.vault, "homelab-vault-archive", "sessions", "2026-07-21-slug.md"
        )
        self.assertEqual(result, expected)

    # --- absolute path -----------------------------------------------------

    def test_absolute_path_unchanged(self):
        abs_path = os.path.join(self.tmp, "elsewhere", "sessions", "x.md")
        result = kicker._resolve_session_note(abs_path)
        self.assertEqual(result, abs_path)

    # --- empty / blank -----------------------------------------------------

    def test_blank_returns_empty_string(self):
        self.assertEqual(kicker._resolve_session_note(""), "")
        self.assertEqual(kicker._resolve_session_note("   "), "")


if __name__ == "__main__":
    unittest.main()

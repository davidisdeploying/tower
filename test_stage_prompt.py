"""Focused tests for vaultsearch.stage_prompt and the write_note raw-write guard
(FLEET-WORKER2-BUILD-20260710-safe-stage-prompt).

Uses a tempdir monkeypatched onto vaultsearch.VAULTS_ROOT — no live vault, no
network, no NAS. Run with:
    python -m unittest test_stage_prompt -v
"""
import os
import shutil
import tempfile
import unittest

import vaultsearch as vs

TOKEN = "FLEET-WORKER2-BUILD-20260710-safe-stage-prompt"
LEGACY_TOKEN = "FLEET-BUILD-20260710-legacy-slug"


class StagePromptTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fleet-stage-prompt-test-")
        self._orig_root = vs.VAULTS_ROOT
        vs.VAULTS_ROOT = os.path.realpath(self.tmp)

    def tearDown(self):
        vs.VAULTS_ROOT = self._orig_root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _full(self, rel_path: str) -> str:
        return os.path.join(vs.VAULTS_ROOT, rel_path)

    # --- valid staging -----------------------------------------------------

    def test_valid_prompt_staging(self):
        content = f"# Build\n\n{TOKEN}\n"
        result = vs.stage_prompt("worker2", "prompts", TOKEN, content)
        self.assertTrue(result["ok"], result)
        expected_path = os.path.join("homelab-vault", "to-worker2", "prompts", "latest.md")
        self.assertEqual(result["path"], expected_path)
        self.assertEqual(result["seat"], "worker2")
        self.assertEqual(result["lane"], "prompts")
        self.assertEqual(result["token"], TOKEN)
        self.assertEqual(result["bytes"], len(content.encode("utf-8")))
        self.assertNotIn("archive_path", result)
        with open(self._full(expected_path), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), content)

    def test_all_five_seats(self):
        for seat in ("delta", "charlie", "localworker", "worker4", "alpha"):
            content = f"seat={seat}\n{TOKEN}\n"
            result = vs.stage_prompt(seat, "prompts", TOKEN, content)
            self.assertTrue(result["ok"], (seat, result))
            self.assertEqual(result["seat"], seat)
            expected = os.path.join("homelab-vault", f"to-{seat}", "prompts", "latest.md")
            self.assertEqual(result["path"], expected)
            self.assertTrue(os.path.isfile(self._full(expected)))

    def test_both_lanes(self):
        for lane in ("prompts", "recon"):
            content = f"lane={lane}\n{TOKEN}\n"
            result = vs.stage_prompt("worker3", lane, TOKEN, content)
            self.assertTrue(result["ok"], (lane, result))
            self.assertEqual(result["lane"], lane)
            expected = os.path.join("homelab-vault", "to-worker3", lane, "latest.md")
            self.assertEqual(result["path"], expected)

    # --- validation failures -------------------------------------------------

    def test_invalid_seat_rejected(self):
        result = vs.stage_prompt("cc", "prompts", TOKEN, f"body {TOKEN}")
        self.assertFalse(result["ok"])
        self.assertIn("seat", result["error"])
        self.assertFalse(os.path.isdir(self._full("homelab-vault")))

    def test_invalid_lane_rejected(self):
        result = vs.stage_prompt("worker2", "runs", TOKEN, f"body {TOKEN}")
        self.assertFalse(result["ok"])
        self.assertIn("lane", result["error"])
        self.assertFalse(os.path.isdir(self._full("homelab-vault")))

    def test_empty_content_rejected(self):
        result = vs.stage_prompt("worker2", "prompts", TOKEN, "")
        self.assertFalse(result["ok"])
        self.assertIn("content", result["error"])

    def test_whitespace_only_content_rejected(self):
        result = vs.stage_prompt("worker2", "prompts", TOKEN, "   \n\t  ")
        self.assertFalse(result["ok"])
        self.assertIn("content", result["error"])

    def test_non_string_content_rejected(self):
        result = vs.stage_prompt("worker2", "prompts", TOKEN, 12345)
        self.assertFalse(result["ok"])
        self.assertIn("content", result["error"])

    def test_malformed_token_rejected(self):
        for bad in ("", "not-a-token", "FLEET-BUILD-notadate-slug", "FLEET-BUILD-20260710-"):
            result = vs.stage_prompt("worker2", "prompts", bad, f"body {bad or 'x'}")
            self.assertFalse(result["ok"], bad)

    def test_token_missing_from_content_rejected(self):
        result = vs.stage_prompt("worker2", "prompts", TOKEN, "this body does not carry it")
        self.assertFalse(result["ok"])
        self.assertIn("token", result["error"])

    def test_error_never_echoes_content(self):
        secret_content = "SECRET-MARKER-should-not-leak-into-error"
        result = vs.stage_prompt("bogus-seat", "prompts", TOKEN, secret_content)
        self.assertFalse(result["ok"])
        self.assertNotIn(secret_content, result["error"])
        result2 = vs.stage_prompt("worker2", "prompts", TOKEN, secret_content)
        self.assertFalse(result2["ok"])
        self.assertNotIn(secret_content, result2["error"])

    def test_current_and_legacy_token_forms(self):
        for tok in (TOKEN, LEGACY_TOKEN, "FLEET-WORKER3-RECON-20260710-recon-slug"):
            result = vs.stage_prompt("worker2", "prompts", tok, f"body {tok}")
            self.assertTrue(result["ok"], (tok, result))

    # --- write behavior --------------------------------------------------

    def test_atomic_replacement_no_tmp_leftover(self):
        vs.stage_prompt("worker2", "prompts", TOKEN, f"first {TOKEN}")
        result = vs.stage_prompt("worker2", "prompts", TOKEN, f"second {TOKEN}")
        self.assertTrue(result["ok"])
        lane_dir = self._full(os.path.join("homelab-vault", "to-worker2", "prompts"))
        names = os.listdir(lane_dir)
        self.assertIn("latest.md", names)
        self.assertFalse(any(n.startswith(".latest.md.tmp") for n in names))
        with open(os.path.join(lane_dir, "latest.md"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), f"second {TOKEN}")

    def test_archive_previous_true_creates_archive_and_preserves_old_content(self):
        old_content = f"old body {TOKEN}"
        new_content = f"new body {TOKEN}"
        vs.stage_prompt("worker2", "prompts", TOKEN, old_content)
        result = vs.stage_prompt("worker2", "prompts", TOKEN, new_content, archive_previous=True)
        self.assertTrue(result["ok"])
        self.assertIn("archive_path", result)
        archive_full = self._full(result["archive_path"])
        self.assertTrue(os.path.isfile(archive_full))
        with open(archive_full, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), old_content)
        latest_full = self._full(result["path"])
        with open(latest_full, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), new_content)

    def test_archive_previous_false_creates_no_archive(self):
        vs.stage_prompt("worker2", "prompts", TOKEN, f"old body {TOKEN}")
        result = vs.stage_prompt("worker2", "prompts", TOKEN, f"new body {TOKEN}", archive_previous=False)
        self.assertTrue(result["ok"])
        self.assertNotIn("archive_path", result)
        archive_dir = self._full(os.path.join("homelab-vault", "to-worker2", "prompts", "archive"))
        self.assertFalse(os.path.isdir(archive_dir))

    def test_no_archive_on_first_write_even_with_archive_true(self):
        result = vs.stage_prompt("worker2", "prompts", TOKEN, f"first {TOKEN}", archive_previous=True)
        self.assertTrue(result["ok"])
        self.assertNotIn("archive_path", result)

    def test_no_writes_outside_canonical_destination(self):
        vs.stage_prompt("worker2", "prompts", TOKEN, f"body {TOKEN}")
        # Only homelab-vault/to-worker2/prompts/{latest.md,archive/} should exist.
        for root, dirs, files in os.walk(vs.VAULTS_ROOT):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, vs.VAULTS_ROOT)
                self.assertEqual(
                    rel,
                    os.path.join("homelab-vault", "to-worker2", "prompts", "latest.md"),
                    f"unexpected file written: {rel}",
                )


class WriteNoteRelayGuardTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fleet-write-note-guard-test-")
        self._orig_root = vs.VAULTS_ROOT
        vs.VAULTS_ROOT = os.path.realpath(self.tmp)

    def tearDown(self):
        vs.VAULTS_ROOT = self._orig_root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_raw_write_rejects_to_star(self):
        result = vs.write_note("homelab-vault/to-worker2/prompts/latest.md", "x")
        self.assertFalse(result["ok"])
        self.assertIn("stage_prompt", result["error"])
        self.assertFalse(
            os.path.exists(os.path.join(vs.VAULTS_ROOT, "homelab-vault/to-worker2/prompts/latest.md"))
        )

    def test_raw_write_rejects_from_star(self):
        result = vs.write_note("homelab-vault/from-worker2/prompts/latest_response.md", "x")
        self.assertFalse(result["ok"])
        self.assertIn("script-owned delivery path", result["error"])
        self.assertFalse(
            os.path.exists(
                os.path.join(vs.VAULTS_ROOT, "homelab-vault/from-worker2/prompts/latest_response.md")
            )
        )

    def test_raw_write_rejects_all_seats_both_lanes(self):
        for seat in ("delta", "charlie", "localworker", "worker4", "alpha"):
            for direction in ("to", "from"):
                for lane in ("prompts", "recon"):
                    path = f"homelab-vault/{direction}-{seat}/{lane}/latest.md"
                    result = vs.write_note(path, "x")
                    self.assertFalse(result["ok"], path)

    def test_ordinary_session_note_write_allowed(self):
        result = vs.write_note("homelab-vault/sessions/2026-07-10-safe-stage-prompt.md", "# note\n")
        self.assertTrue(result["ok"], result)

    def test_ordinary_audits_write_allowed(self):
        result = vs.write_note("homelab-vault/audits/2026-07-10-check.md", "# audit\n")
        self.assertTrue(result["ok"], result)

    def test_sibling_vault_writes_allowed(self):
        for path in ("loupe-vault/DECISIONS.md", "homelab-vault/notes/foo.md"):
            result = vs.write_note(path, "content\n", mode="append")
            self.assertTrue(result["ok"], (path, result))

    def test_readback_search_and_write_behavior_unaffected(self):
        # write_note for a non-relay path still round-trips normally.
        result = vs.write_note("loupe-vault/scratch.md", "hello\n")
        self.assertTrue(result["ok"])
        self.assertEqual(vs.read_note("loupe-vault/scratch.md"), "hello\n")


if __name__ == "__main__":
    unittest.main()

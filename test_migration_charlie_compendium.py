"""Migration assertion for the loupe-vault -> homelab-vault relay-root
cutover (FLEET-WORKER2-BUILD-20260710-homelab-vault-migration).

Guards against a regression that quietly repoints any relay path back under
loupe-vault/to-* or loupe-vault/from-* (source-string scan) or that leaves
kicker.VAULT (and its derived per-seat run/transcript roots) pointed at the
old vault. Read-only, no filesystem mutation. Run with:
    python -m unittest test_migration_charlie_compendium -v
"""
import inspect
import os
import unittest

import kicker


class RelayRootMigrationTestCase(unittest.TestCase):
    def test_vault_is_charlie_compendium(self):
        self.assertTrue(
            kicker.VAULT.endswith(os.path.join("Vaults", "homelab-vault")),
            f"kicker.VAULT does not point at homelab-vault: {kicker.VAULT}",
        )
        self.assertNotIn("loupe-vault", kicker.VAULT)

    def test_seat_run_and_transcript_roots_under_charlie_compendium(self):
        for name in (
            "WORKER1_RUNS", "DELTA_TRANSCRIPTS",
            "WORKER3_RUNS", "CHARLIE_TRANSCRIPTS",
            "LOCALWORKER_RUNS", "LOCALWORKER_TRANSCRIPTS",
            "WORKER4_RUNS", "WORKER4_TRANSCRIPTS",
            "WORKER2_RUNS", "ALPHA_TRANSCRIPTS",
        ):
            path = getattr(kicker, name)
            self.assertIn(
                "homelab-vault", path, f"kicker.{name} not under homelab-vault: {path}"
            )
            self.assertNotIn(
                "loupe-vault", path, f"kicker.{name} still under loupe-vault: {path}"
            )

    def test_no_source_reference_to_loupe_vault_relay_queues(self):
        """No active kicker.py code path should build a to-*/from-* relay path
        under loupe-vault. Loupe-project-knowledge comments (e.g. "Loupe
        project knowledge lives in loupe-vault") are fine; a literal
        'loupe-vault/to-' or 'loupe-vault/from-' path fragment is not.
        """
        src = inspect.getsource(kicker)
        self.assertNotIn("loupe-vault/to-", src)
        self.assertNotIn("loupe-vault/from-", src)
        self.assertNotIn('"Vaults", "loupe-vault"', src)

    def test_charlie_compendium_relay_tree_exists_on_disk(self):
        """Deployment check, not a code check: only meaningful on a host that
        actually carries the relay tree. Skips cleanly anywhere else."""
        home = os.path.expanduser("~")
        root = os.path.join(home, "Vaults", "homelab-vault")
        if not os.path.isdir(root):
            self.skipTest(f"no relay root on this host: {root}")
        # Seat names are deployment configuration. Take them from the module
        # under test rather than hard-coding one host's list, and skip where
        # this machine is not the deployment those names describe.
        seats = sorted({
            os.path.basename(os.path.dirname(getattr(kicker, name)))[len("from-"):]
            for name in dir(kicker)
            if name.endswith("_RUNS")
            and os.path.basename(os.path.dirname(getattr(kicker, name))).startswith("from-")
        })
        present = [s for s in seats if os.path.isdir(os.path.join(root, f"from-{s}"))]
        if not present:
            self.skipTest(f"no configured relay queues under {root}")
        for seat in present:
            self.assertTrue(
                os.path.isdir(os.path.join(root, f"to-{seat}")),
                f"from-{seat} has no matching to-{seat} queue",
            )


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import json
import tempfile
import unittest

from scripts import control_plane_offhost_sync as syncer


class ControlPlaneOffhostSyncTests(unittest.TestCase):
    def make_snapshot(self, root: Path, stamp: str = "20260728T203000Z") -> Path:
        snapshot = root / stamp
        snapshot.mkdir(parents=True)
        payload = snapshot / "events.sqlite3"
        payload.write_bytes(b"bounded-test-payload")
        manifest = {
            "schema": "tower-control-plane-backup-v1",
            "records": [
                {
                    "backup": payload.name,
                    "sha256": syncer.sha256(payload),
                }
            ],
        }
        (snapshot / "manifest.json").write_text(json.dumps(manifest) + "\n")
        (root / "latest").symlink_to(stamp)
        return snapshot

    def test_validate_snapshot_accepts_verified_latest(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            snapshot = self.make_snapshot(root)
            actual, digest = syncer.validate_snapshot(root)
            self.assertEqual(actual, snapshot.resolve())
            self.assertEqual(digest, syncer.sha256(snapshot / "manifest.json"))

    def test_validate_snapshot_rejects_changed_payload(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            snapshot = self.make_snapshot(root)
            (snapshot / "events.sqlite3").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "source verification failed"):
                syncer.validate_snapshot(root)

    def test_validate_snapshot_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp_name:
            base = Path(temp_name)
            root = base / "root"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "latest").symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "escapes snapshot root"):
                syncer.validate_snapshot(root)


if __name__ == "__main__":
    unittest.main()

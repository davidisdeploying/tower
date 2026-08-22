import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import vaultsearch as vs


class IndexMetadataTestCase(unittest.TestCase):
    def _database(self, root, with_chunks=True):
        path = os.path.join(root, "vault.db")
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE notes(id INTEGER PRIMARY KEY, indexed_at TEXT)")
        db.executemany(
            "INSERT INTO notes(indexed_at) VALUES (?)",
            [("2026-07-20T01:02:03",), ("2026-07-22T04:05:06",)],
        )
        if with_chunks:
            db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY)")
            db.executemany("INSERT INTO chunks(id) VALUES (?)", [(1,), (2,), (3,)])
        db.execute("PRAGMA user_version=7")
        db.commit()
        db.close()
        return path

    def test_reports_stable_counts_and_honest_timestamp_source(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._database(root)
            with mock.patch.object(vs, "INDEX_PATH", path), mock.patch.object(vs.sqlite_vec, "load"):
                result = vs.index_metadata()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["document_count"], 2)
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual(result["schema_version"], 7)
        self.assertGreater(result["database_bytes"], 0)
        self.assertEqual(result["published_at_source"], "database_file_mtime")
        self.assertFalse(result["authoritative_build_timestamp_available"])
        self.assertEqual(result["oldest_document_indexed_at"], "2026-07-20T01:02:03")
        self.assertEqual(result["latest_document_indexed_at"], "2026-07-22T04:05:06")

    def test_missing_chunks_table_reports_zero(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._database(root, with_chunks=False)
            with mock.patch.object(vs, "INDEX_PATH", path), mock.patch.object(vs.sqlite_vec, "load"):
                result = vs.index_metadata()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["chunk_count"], 0)

    def test_missing_and_corrupt_index_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            missing = os.path.join(root, "missing.db")
            with mock.patch.object(vs, "INDEX_PATH", missing):
                self.assertFalse(vs.index_metadata()["ok"])
            corrupt = os.path.join(root, "corrupt.db")
            with open(corrupt, "wb") as handle:
                handle.write(b"not sqlite")
            with mock.patch.object(vs, "INDEX_PATH", corrupt), mock.patch.object(vs.sqlite_vec, "load"):
                self.assertFalse(vs.index_metadata()["ok"])

    def test_server_wrapper_delegates(self):
        import server
        expected = {"ok": True, "document_count": 9}
        with mock.patch.object(server.vs, "index_metadata", return_value=expected) as metadata:
            self.assertEqual(server.index_metadata(), expected)
        metadata.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

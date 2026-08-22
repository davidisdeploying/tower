"""Synthetic tests for Tower's secondary transcript-evidence search."""
import json
import os
import shutil
import sqlite3
import tempfile
import unittest

import numpy as np
import sqlite_vec

import vaultsearch as vs


DIM = 768


def vec(similarity):
    value = np.zeros(DIM, dtype=np.float32)
    value[0] = similarity
    value[1] = np.sqrt(max(0.0, 1.0 - similarity * similarity))
    return value


class HistorySearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tower-history-")
        self.db_path = os.path.join(self.tmp, "vault.db")
        self.vaults = os.path.join(self.tmp, "Vaults")
        evidence = (
            "homelab-vault/files/conversations/evidence/macbook/codex/2026/"
            "abc.jsonl.gz"
        )
        manifest = (
            "homelab-vault/files/conversations/manifests/macbook/codex/2026/"
            "abc.json"
        )
        os.makedirs(os.path.dirname(os.path.join(self.vaults, manifest)), exist_ok=True)
        with open(os.path.join(self.vaults, manifest), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "raw_path": "homelab-vault/files/conversations/raw/macbook/codex/2026/abc.jsonl.gz",
                    "card_path": "homelab-vault/conversation-index/2026/07/codex-abc.md",
                },
                handle,
            )

        db = sqlite3.connect(self.db_path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.execute(
            "CREATE TABLE transcript_docs(id INTEGER PRIMARY KEY,path TEXT UNIQUE,"
            "vault TEXT,conversation_id TEXT,surface TEXT,host TEXT,sha256 TEXT,"
            "mtime REAL,chunk_count INTEGER,indexed_at TEXT)"
        )
        db.execute(
            "CREATE TABLE transcript_chunks(id INTEGER PRIMARY KEY,doc_id INTEGER,"
            "ordinal INTEGER,role TEXT,source_line INTEGER,timestamp TEXT,text TEXT,"
            "sha256 TEXT,indexed_at TEXT)"
        )
        db.execute(
            "CREATE VIRTUAL TABLE vec_transcript_chunks USING "
            "vec0(embedding float[768] distance_metric=cosine)"
        )
        db.execute(
            "INSERT INTO transcript_docs VALUES(1,?,?,?,?,?,'x',0,2,'x')",
            (evidence, "homelab-vault", "abc", "codex", "macbook"),
        )
        rows = [
            (1, 1, 0, "user", 4, "2026-07-27T00:00:00Z", "asked about transcript recall", 0.92),
            (2, 1, 1, "assistant", 5, "2026-07-27T00:00:01Z", "implemented bounded history search", 0.75),
        ]
        for row_id, doc_id, ordinal, role, line, stamp, text, similarity in rows:
            db.execute(
                "INSERT INTO transcript_chunks VALUES(?,?,?,?,?,?,?,'x','x')",
                (row_id, doc_id, ordinal, role, line, stamp, text),
            )
            db.execute(
                "INSERT INTO vec_transcript_chunks(rowid,embedding) VALUES(?,?)",
                (row_id, vec(similarity).tobytes()),
            )
        db.commit()
        db.close()

        self.old_index = vs.INDEX_PATH
        self.old_root = vs.VAULTS_ROOT
        self.old_embed = vs.embed_query
        vs.INDEX_PATH = self.db_path
        vs.VAULTS_ROOT = os.path.realpath(self.vaults)
        vs.embed_query = lambda _text: vec(1.0)

    def tearDown(self):
        vs.INDEX_PATH = self.old_index
        vs.VAULTS_ROOT = self.old_root
        vs.embed_query = self.old_embed
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_history_returns_pointers_and_metadata(self):
        rows = vs.search_history("history search", k=1, vault="homelab-vault")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["conversation_id"], "abc")
        self.assertEqual(rows[0]["surface"], "codex")
        self.assertTrue(rows[0]["raw_path"].endswith("abc.jsonl.gz"))
        self.assertIn("transcript recall", rows[0]["excerpt"])

    def test_read_history_is_bounded_and_pageable(self):
        evidence = (
            "homelab-vault/files/conversations/evidence/macbook/codex/2026/"
            "abc.jsonl.gz"
        )
        result = vs.read_history(evidence, start_chunk=1, limit=1, max_chars=12)
        self.assertTrue(result["ok"])
        self.assertEqual(result["returned_chunks"], 1)
        self.assertEqual(result["chunks"][0]["chunk_ordinal"], 1)
        self.assertTrue(result["chunks"][0]["truncated"])
        self.assertEqual(result["returned_chars"], 12)

    def test_read_history_rejects_traversal(self):
        result = vs.read_history("../secret", 0, 1)
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()

"""Focused tests for vaultsearch.search's bounded hybrid exact-reference
search (FLEET-WORKER2-BUILD-20260722-tower-bounded-hybrid-search).

Covers: k conversion/clamping to [1, 50], exact full-path/basename/title
promotion ahead of semantic results, deterministic exact ordering,
exact+semantic dedup by canonical base note path (incl. chunk anchors),
LIKE-escape correctness for basename matching, rejection of absolute /
traversal-like / control-character / wrong-case references, and that
ordinary semantic queries are otherwise unaffected.

Fully synthetic sqlite-vec DB (tempfile) + a fixed fake embed_query — no
live index/vault.db, no ~/Vaults, no ONNX/tokenizer model, no network, no
GPU, no service. Run with:
    python -m unittest test_search_exact_reference -v
"""
import os
import shutil
import sqlite3
import tempfile
import unittest

import numpy as np
import sqlite_vec

import vaultsearch as vs

DIM = 768


def _vec(similarity: float) -> np.ndarray:
    """Deterministic unit delta whose cosine similarity to the fixed query
    vector _vec(1.0) is exactly `similarity` (query vector = e0)."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = similarity
    remainder = max(1.0 - similarity * similarity, 0.0)
    v[1] = np.sqrt(remainder, dtype=np.float32)
    return v.astype(np.float32)


QUERY_VEC = _vec(1.0)

# note_id -> (path, vault, title, vec_notes_similarity_or_None)
NOTES = [
    (1, "root.md", "testvault", "Root Note", 0.9),
    (2, "dir/sub/root.md", "testvault", "Nested Root", 0.05),
    (3, "titled.md", "testvault", "Exact Title Match", 0.2),
    (4, "chunked.md", "testvault", "Chunked Note", 0.5),
    (5, "dup.md", "testvault", "Duplicate Note", 0.85),
    (6, "under_score/a_note.md", "testvault", "Underscore Note", 0.4),
    (7, "other/aZnote.md", "testvault", "Decoy Note", 0.15),
    (8, "orphan.md", "testvault", "Orphan Note", None),
]

# (chunk_id, note_id, anchor, text, similarity)
CHUNKS = [
    (1, 4, "h1", "chunk one text", 0.7),
    (2, 4, "h2", "chunk two text", 0.3),
]

NOTE_FILE_TEXT = {
    "root.md": "root note body\n",
    "dir/sub/root.md": "nested root body\n",
    "titled.md": "titled note body\n",
    "chunked.md": "chunked note body\n",
    "dup.md": "dup note body\n",
    "under_score/a_note.md": "underscore note body\n",
    "other/aZnote.md": "decoy note body\n",
    "orphan.md": "orphan note body\n",
}


def _build_index_db(db_path: str) -> None:
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute(
        """CREATE TABLE notes(
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            vault TEXT,
            sha256 TEXT,
            mtime REAL,
            char_len INTEGER,
            token_len INTEGER,
            title TEXT,
            indexed_at TEXT)"""
    )
    db.execute(
        """CREATE TABLE chunks(
            id INTEGER PRIMARY KEY,
            note_id INTEGER REFERENCES notes(id),
            anchor TEXT,
            ordinal INTEGER,
            char_len INTEGER,
            token_len INTEGER,
            sha256 TEXT,
            text TEXT,
            indexed_at TEXT,
            UNIQUE(note_id, ordinal))"""
    )
    db.execute(
        "CREATE VIRTUAL TABLE vec_notes USING vec0(embedding float[768] distance_metric=cosine)"
    )
    db.execute(
        "CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[768] distance_metric=cosine)"
    )

    for note_id, path, vault, title, sim in NOTES:
        db.execute(
            "INSERT INTO notes(id, path, vault, sha256, mtime, char_len, token_len, title, indexed_at)"
            " VALUES (?, ?, ?, 'x', 0.0, 1, 1, ?, 'x')",
            (note_id, path, vault, title),
        )
        if sim is not None:
            db.execute(
                "INSERT INTO vec_notes(rowid, embedding) VALUES (?, ?)",
                (note_id, _vec(sim).tobytes()),
            )

    for chunk_id, note_id, anchor, text, sim in CHUNKS:
        db.execute(
            "INSERT INTO chunks(id, note_id, anchor, ordinal, char_len, token_len, sha256, text, indexed_at)"
            " VALUES (?, ?, ?, ?, 1, 1, 'x', ?, 'x')",
            (chunk_id, note_id, anchor, chunk_id - 1, text),
        )
        db.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (chunk_id, _vec(sim).tobytes()),
        )

    db.commit()
    db.close()


class ExactReferenceSearchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="fleet-search-exact-test-")
        self.db_path = os.path.join(self.tmpdir, "vault.db")
        _build_index_db(self.db_path)

        self.vaults_root = os.path.join(self.tmpdir, "vaults")
        for rel, text in NOTE_FILE_TEXT.items():
            full = os.path.join(self.vaults_root, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(text)

        self._orig_index_path = vs.INDEX_PATH
        self._orig_vaults_root = vs.VAULTS_ROOT
        self._orig_embed_query = vs.embed_query
        self._open_index_calls = []
        self._orig_open_index = vs.open_index

        vs.INDEX_PATH = self.db_path
        vs.VAULTS_ROOT = os.path.realpath(self.vaults_root)
        vs.embed_query = lambda text: QUERY_VEC

        def _counting_open_index():
            self._open_index_calls.append(1)
            return self._orig_open_index()

        vs.open_index = _counting_open_index

    def tearDown(self):
        vs.INDEX_PATH = self._orig_index_path
        vs.VAULTS_ROOT = self._orig_vaults_root
        vs.embed_query = self._orig_embed_query
        vs.open_index = self._orig_open_index
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---- k clamping (unit, via _clamp_k) ----

    def test_clamp_k_default_and_in_range(self):
        self.assertEqual(vs._clamp_k(8), 8)
        self.assertEqual(vs._clamp_k(1), 1)
        self.assertEqual(vs._clamp_k(50), 50)

    def test_clamp_k_below_range(self):
        self.assertEqual(vs._clamp_k(-1), 1)
        self.assertEqual(vs._clamp_k(0), 1)

    def test_clamp_k_above_range(self):
        self.assertEqual(vs._clamp_k(51), 50)
        self.assertEqual(vs._clamp_k(4096), 50)
        self.assertEqual(vs._clamp_k(1000000), 50)

    def test_clamp_k_int_conversion(self):
        self.assertEqual(vs._clamp_k("8"), 8)
        with self.assertRaises(ValueError):
            vs._clamp_k("not-a-number")
        with self.assertRaises(TypeError):
            vs._clamp_k(None)

    # ---- escape helper (unit) ----

    def test_escape_like_escapes_wildcards_and_backslash(self):
        self.assertEqual(vs._escape_like("a_b"), "a\\_b")
        self.assertEqual(vs._escape_like("a%b"), "a\\%b")
        self.assertEqual(vs._escape_like("a\\b"), "a\\\\b")
        self.assertEqual(vs._escape_like("a_b%c\\d"), "a\\_b\\%c\\\\d")

    # ---- default k / omitted ----

    def test_default_k_is_8(self):
        results = vs.search("no such match anywhere")
        self.assertLessEqual(len(results), 8)

    # ---- exact full path promotion ----

    def test_exact_full_path_promoted_first(self):
        results = vs.search("dir/sub/root.md", k=8)
        self.assertTrue(results)
        self.assertEqual(results[0]["path"], "dir/sub/root.md")
        self.assertEqual(results[0]["score"], 1.0)

    # ---- exact title equality ----

    def test_exact_title_promoted_first(self):
        results = vs.search("Exact Title Match", k=8)
        self.assertTrue(results)
        self.assertEqual(results[0]["path"], "titled.md")
        self.assertEqual(results[0]["score"], 1.0)

    # ---- ambiguous basename ----

    def test_ambiguous_basename_returns_all_matches_in_deterministic_order(self):
        results = vs.search("root.md", k=8)
        paths = [r["path"] for r in results if r["score"] == 1.0]
        # root.md (full path / root-level match) sorts before dir/sub/root.md
        # (basename-tier match), matching canonical-path ordering.
        self.assertEqual(paths, ["root.md", "dir/sub/root.md"])

    def test_ambiguous_basename_respects_k(self):
        results = vs.search("root.md", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["path"], "root.md")

    # ---- exact+semantic duplicate collapse ----

    def test_exact_and_semantic_duplicate_collapses_to_one(self):
        results = vs.search("dup.md", k=8)
        matches = [r for r in results if r["path"] == "dup.md"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["score"], 1.0)

    # ---- chunk anchor + exact base path dedup ----

    def test_chunk_anchor_and_exact_base_path_dedup(self):
        results = vs.search("chunked.md", k=8)
        base_paths = [r["path"].split("#", 1)[0] for r in results]
        self.assertEqual(base_paths.count("chunked.md"), 1)
        self.assertEqual(results[0]["path"], "chunked.md")
        self.assertEqual(results[0]["score"], 1.0)

    def test_chunked_whole_note_suppressed_in_ordinary_semantic_query(self):
        results = vs.search("no such match anywhere", k=8)
        paths = [r["path"] for r in results]
        self.assertNotIn("chunked.md", paths)
        self.assertIn("chunked.md#h1", paths)
        self.assertIn("chunked.md#h2", paths)

    # ---- ordinary semantic query preserves order ----

    def test_ordinary_semantic_query_order(self):
        results = vs.search("no such match anywhere", k=8)
        paths = [r["path"] for r in results]
        self.assertEqual(
            paths,
            [
                "root.md",
                "dup.md",
                "chunked.md#h1",
                "under_score/a_note.md",
                "chunked.md#h2",
                "titled.md",
                "other/aZnote.md",
                "dir/sub/root.md",
            ],
        )
        for r in results:
            self.assertNotEqual(r["score"], 1.0)

    def test_ordinary_semantic_query_respects_k(self):
        results = vs.search("no such match anywhere", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["path"], "root.md")

    # ---- k boundary values end-to-end ----

    def test_k_minus_one_and_zero_clamp_to_one(self):
        for k in (-1, 0):
            results = vs.search("no such match anywhere", k=k)
            self.assertEqual(len(results), 1)

    def test_k_51_4096_1000000_clamp_to_50(self):
        for k in (51, 4096, 1000000):
            results = vs.search("no such match anywhere", k=k)
            self.assertLessEqual(len(results), 50)

    def test_final_len_never_exceeds_clamped_k(self):
        for k in (1, 3, 8, 50, 1000000):
            results = vs.search("no such match anywhere", k=k)
            expected_cap = vs._clamp_k(k)
            self.assertLessEqual(len(results), expected_cap)

    # ---- LIKE escape correctness end-to-end ----

    def test_literal_underscore_does_not_wildcard_expand(self):
        results = vs.search("a_note.md", k=8)
        paths = [r["path"] for r in results if r["score"] == 1.0]
        self.assertEqual(paths, ["under_score/a_note.md"])
        self.assertNotIn("other/aZnote.md", paths)

    # ---- rejected reference shapes: no exact match, no crash ----

    def test_absolute_path_does_not_exact_match(self):
        results = vs.search("/root.md", k=8)
        exact = [r for r in results if r["score"] == 1.0]
        self.assertEqual(exact, [])

    def test_traversal_like_path_does_not_exact_match(self):
        results = vs.search("dir/../root.md", k=8)
        exact = [r for r in results if r["score"] == 1.0]
        self.assertEqual(exact, [])

    def test_control_character_path_does_not_exact_match(self):
        results = vs.search("roo\x07t.md", k=8)
        exact = [r for r in results if r["score"] == 1.0]
        self.assertEqual(exact, [])

    def test_wrong_case_path_does_not_exact_match(self):
        results = vs.search("Root.md", k=8)
        exact = [r for r in results if r["score"] == 1.0 and r["path"] == "root.md"]
        self.assertEqual(exact, [])

    # ---- exact-only note with no delta rows ----

    def test_exact_only_note_without_vectors_surfaces_by_path(self):
        results = vs.search("orphan.md", k=8)
        matches = [r for r in results if r["path"] == "orphan.md"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["score"], 1.0)

    def test_exact_only_note_without_vectors_surfaces_by_title(self):
        results = vs.search("Orphan Note", k=8)
        matches = [r for r in results if r["path"] == "orphan.md"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["score"], 1.0)

    # ---- response shape ----

    def test_every_result_has_exactly_five_keys(self):
        for query in ("root.md", "no such match anywhere", "dup.md"):
            for r in vs.search(query, k=8):
                self.assertEqual(
                    set(r.keys()), {"path", "vault", "title", "score", "excerpt"}
                )

    # ---- single connection reuse ----

    def test_single_open_index_call_per_search(self):
        vs.search("root.md", k=8)
        self.assertEqual(len(self._open_index_calls), 1)


if __name__ == "__main__":
    unittest.main()

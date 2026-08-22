"""Vault search core: pinned nomic query-embed (CPU) + cosine KNN over the
transported sqlite-vec index. Shared by the parity probe and the MCP server.

The embed recipe here MUST stay byte-for-byte identical to charlie's index side
or the query vectors land in a different space and parity breaks:
  "search_query: " + text
  -> tokenizers (tokenizer.json), truncation max_length 8192
  -> int64 input_ids / attention_mask / token_type_ids(=zeros)
  -> onnxruntime CPUExecutionProvider, last_hidden_state [B,T,768]
  -> attention-mask MEAN pool: sum(h*mask)/clip(sum(mask), 1e-9)
  -> L2-normalize (eps 1e-12) -> 768-d float32
"""
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import sqlite_vec
import onnxruntime as ort
from tokenizers import Tokenizer

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "tower")
INDEX_PATH = (
    os.environ.get("TOWER_INDEX_PATH")
    or os.environ.get("FLEET_TOWER_INDEX_PATH")
    or os.path.expanduser("~/.local/share/tower/index/vault.db")
)
MODEL_PATH = os.path.join(ROOT, "model", "onnx", "model.onnx")
TOKENIZER_PATH = os.path.join(ROOT, "model", "tokenizer.json")
VAULTS_ROOT = os.path.realpath(os.path.join(HOME, "Vaults"))

MAX_LEN = 8192

_tokenizer = None
_session = None


def _lazy_load():
    global _tokenizer, _session
    if _tokenizer is None:
        tok = Tokenizer.from_file(TOKENIZER_PATH)
        tok.enable_truncation(max_length=MAX_LEN)
        _tokenizer = tok
    if _session is None:
        so = ort.SessionOptions()
        _session = ort.InferenceSession(
            MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"]
        )
    return _tokenizer, _session


def embed_query(text: str) -> np.ndarray:
    """Pinned query-embed -> 768-d float32 L2-normalized vector."""
    tok, sess = _lazy_load()
    enc = tok.encode("search_query: " + text)
    ids = np.asarray([enc.ids], dtype=np.int64)
    mask = np.asarray([enc.attention_mask], dtype=np.int64)
    ttype = np.zeros_like(ids, dtype=np.int64)
    feeds = {
        "input_ids": ids,
        "attention_mask": mask,
        "token_type_ids": ttype,
    }
    # Only feed inputs the model actually declares (some exports omit token_type_ids).
    want = {i.name for i in sess.get_inputs()}
    feeds = {k: v for k, v in feeds.items() if k in want}
    out = sess.run(None, feeds)[0]  # [1, T, 768]
    h = out[0].astype(np.float32)  # [T, 768]
    m = mask[0].astype(np.float32)[:, None]  # [T, 1]
    pooled = (h * m).sum(axis=0) / np.clip(m.sum(axis=0), 1e-9, None)
    norm = np.linalg.norm(pooled)
    pooled = pooled / max(norm, 1e-12)
    return pooled.astype(np.float32)


def open_index():
    db = sqlite3.connect(f"file://{INDEX_PATH}?mode=ro", uri=True)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def _has_chunks_table(db) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
    ).fetchone()
    return row is not None


def _has_transcript_table(db) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_transcript_chunks'"
    ).fetchone()
    return row is not None


def index_metadata() -> dict:
    """Return a stable, read-only freshness snapshot of the transported index.

    The current index schema has no authoritative build-metadata table, so the
    publication timestamp is explicitly sourced from the database file mtime.
    Per-document indexed_at bounds are reported separately and never conflated
    with that publication timestamp.
    """
    try:
        before = os.stat(INDEX_PATH)
        if not stat.S_ISREG(before.st_mode):
            return {"ok": False, "error": "index is not a regular file"}
    except OSError as exc:
        return {"ok": False, "error": f"index unavailable: {type(exc).__name__}"}

    db = None
    try:
        db = open_index()
        document_count, oldest_indexed_at, latest_indexed_at = db.execute(
            "SELECT COUNT(*), MIN(indexed_at), MAX(indexed_at) FROM notes"
        ).fetchone()
        chunks_present = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone() is not None
        chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if chunks_present else 0
        transcript_present = _has_transcript_table(db)
        transcript_document_count = (
            db.execute("SELECT COUNT(*) FROM transcript_docs").fetchone()[0]
            if transcript_present else 0
        )
        transcript_chunk_count = (
            db.execute("SELECT COUNT(*) FROM transcript_chunks").fetchone()[0]
            if transcript_present else 0
        )
        schema_version = db.execute("PRAGMA user_version").fetchone()[0]
    except (OSError, sqlite3.Error) as exc:
        return {"ok": False, "error": f"index metadata unavailable: {type(exc).__name__}"}
    finally:
        if db is not None:
            db.close()

    try:
        after = os.stat(INDEX_PATH)
    except OSError as exc:
        return {"ok": False, "error": f"index changed during inspection: {type(exc).__name__}"}
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        return {"ok": False, "error": "index changed during inspection"}

    published_at = datetime.fromtimestamp(before.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ok": True,
        "index_path": os.path.relpath(INDEX_PATH, ROOT),
        "document_count": int(document_count),
        "chunk_count": int(chunk_count),
        "transcript_document_count": int(transcript_document_count),
        "transcript_chunk_count": int(transcript_chunk_count),
        "schema_version": int(schema_version),
        "database_bytes": before.st_size,
        "database_mtime_ns": before.st_mtime_ns,
        "published_at": published_at,
        "published_at_source": "database_file_mtime",
        "authoritative_build_timestamp_available": False,
        "oldest_document_indexed_at": oldest_indexed_at,
        "latest_document_indexed_at": latest_indexed_at,
    }


def _clamp_k(k) -> int:
    """int(k) then clamp to [1, 50]. Malformed input raises ValueError/TypeError."""
    k = int(k)
    if k < 1:
        return 1
    if k > 50:
        return 50
    return k


# Conservative vault-relative Markdown reference: no leading slash, no
# backslash, no control chars (the character class already excludes both),
# '.' segments allowed but '..' and empty segments are rejected explicitly.
_MD_REF_RE = re.compile(r"^[A-Za-z0-9_. /-]+\.md$")


def _parse_md_reference(stripped: str):
    """Return `stripped` if it is a safe .md path reference, else None."""
    if not stripped or "\\" in stripped:
        return None
    if any(ord(c) < 32 or ord(c) == 127 for c in stripped):
        return None
    if not _MD_REF_RE.match(stripped):
        return None
    if stripped.startswith("/"):
        return None
    parts = stripped.split("/")
    if any(p == "" or p == ".." for p in parts):
        return None
    return stripped


def _escape_like(s: str) -> str:
    """Escape a literal string for use in a LIKE ... ESCAPE '\\' pattern."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _exact_candidates(db, query: str, excerpt_chars: int) -> list:
    """Exact-reference candidates via the SAME connection `search()` opened.

    Priority order: full path match, then exact title equality (sorted by
    canonical path), then basename-only reference matches (sorted by
    canonical path). Deduplicated by canonical notes.path. Equality only —
    never a contains/fuzzy scan.
    """
    stripped = query.strip()
    if not stripped:
        return []

    full_path_rows = []
    basename_rows = []

    ref = _parse_md_reference(stripped)
    if ref is not None:
        if "/" in ref:
            full_path_rows = db.execute(
                "SELECT path, vault, title FROM notes WHERE path = ?", (ref,)
            ).fetchall()
        else:
            pattern = "%/" + _escape_like(ref)
            rows = db.execute(
                "SELECT path, vault, title FROM notes WHERE path = ? OR path LIKE ? ESCAPE '\\'",
                (ref, pattern),
            ).fetchall()
            for row in rows:
                (full_path_rows if row[0] == ref else basename_rows).append(row)

    title_rows = db.execute(
        "SELECT path, vault, title FROM notes WHERE title = ?", (stripped,)
    ).fetchall()

    ordered = []
    seen = set()
    for tier in (full_path_rows, title_rows, basename_rows):
        for path, vault, title in sorted(set(tier), key=lambda r: r[0]):
            if path in seen:
                continue
            seen.add(path)
            ordered.append((path, vault, title))

    results = []
    for path, vault, title in ordered:
        eff_title = title or (os.path.splitext(os.path.basename(path))[0])
        results.append(
            {
                "path": path,
                "vault": vault,
                "title": eff_title,
                "score": 1.0,
                "excerpt": _read_excerpt(path, excerpt_chars),
            }
        )
    return results


def search(query: str, k: int = 8, excerpt_chars: int = 600):
    """Top-k hybrid search. Returns list of dicts {path, vault, title, score, excerpt}.

    k is int()-converted then clamped to [1, 50] before any SQL (default 8
    unaffected). Exact-reference candidates (full/basename .md path equality
    or exact title equality against `notes`, via the same read-only
    connection as the vector search) are merged FIRST with score 1.0, ahead
    of the semantic KNN results, deduplicated by canonical base note path.

    The semantic side queries vec_notes (whole-note vectors) and, when
    present, vec_chunks (per-chunk vectors) in the same 768-d cosine space,
    merges by distance, and for any note that has chunk rows suppresses its
    whole-note hit in favor of its chunk hit(s) — 3a's chunk index is
    additive (a chunked note keeps its original vec_notes row too), so
    without this a ledger file would double-report as both a truncated
    whole-note excerpt and a chunk.
    """
    k = _clamp_k(k)
    qv = embed_query(query)
    db = open_index()
    try:
        exact = _exact_candidates(db, query, excerpt_chars)

        note_rows = db.execute(
            """
            SELECT n.path, n.vault, n.title, v.distance
            FROM vec_notes v
            JOIN notes n ON n.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (qv.tobytes(), k),
        ).fetchall()

        chunk_rows = []
        chunked_paths = set()
        if _has_chunks_table(db):
            chunk_rows = db.execute(
                """
                SELECT n.path, n.vault, n.title, c.anchor, c.text, v.distance
                FROM vec_chunks v
                JOIN chunks c ON c.id = v.rowid
                JOIN notes n ON n.id = c.note_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (qv.tobytes(), k),
            ).fetchall()
            # Any note with chunk rows at all (not just among these hits) is
            # chunked -- suppress its whole-note vec_notes hit unconditionally.
            chunked_paths = {
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT n.path FROM chunks c JOIN notes n ON n.id = c.note_id"
                ).fetchall()
            }
    finally:
        db.close()

    merged = []
    for path, vault, title, distance in note_rows:
        if path in chunked_paths:
            continue
        eff_title = title or (os.path.splitext(os.path.basename(path))[0])
        merged.append(
            {
                "path": path,
                "vault": vault,
                "title": eff_title,
                "distance": float(distance),
                "excerpt": _read_excerpt(path, excerpt_chars),
            }
        )
    for path, vault, title, anchor, text, distance in chunk_rows:
        eff_title = title or (os.path.splitext(os.path.basename(path))[0])
        merged.append(
            {
                "path": f"{path}#{anchor}",
                "vault": vault,
                "title": eff_title,
                "distance": float(distance),
                "excerpt": text.strip(),
            }
        )

    merged.sort(key=lambda r: r["distance"])
    semantic = []
    for r in merged:
        score = 1.0 - r["distance"]  # cosine distance -> cosine similarity
        semantic.append(
            {
                "path": r["path"],
                "vault": r["vault"],
                "title": r["title"],
                "score": round(score, 4),
                "excerpt": r["excerpt"],
            }
        )

    # Hybrid merge: exact first, then semantic, deduped by canonical base
    # note path (strip #anchor) so a note can't appear once exact and again
    # semantic. Semantic-internal duplicates (e.g. two chunks of one note)
    # are untouched -- only collisions with an exact hit are dropped, so
    # ordinary queries with no exact match are byte-for-byte unaffected.
    exact_paths = {item["path"] for item in exact}
    results = list(exact)
    for item in semantic:
        canon = item["path"].split("#", 1)[0]
        if canon in exact_paths:
            continue
        results.append(item)

    return results[:k]


def _history_manifest_path(evidence_path: str) -> str:
    marker = f"{os.sep}evidence{os.sep}"
    normalized = evidence_path.replace("/", os.sep)
    if marker not in normalized or not normalized.endswith(".jsonl.gz"):
        return ""
    manifest = normalized.replace(marker, f"{os.sep}manifests{os.sep}", 1)
    manifest = manifest[: -len(".jsonl.gz")] + ".json"
    return manifest.replace(os.sep, "/")


def _history_manifest(evidence_path: str) -> dict:
    rel = _history_manifest_path(evidence_path)
    if not rel:
        return {}
    try:
        full = _resolve(rel)
        with open(full, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {}


def search_history(
    query: str,
    k: int = 8,
    vault: str = "",
    surface: str = "",
    host: str = "",
) -> list[dict]:
    """Semantic search over the secondary transcript-evidence collection.

    Only short redacted excerpts are returned. The immutable raw transcript
    remains outside the vector DB and is referenced through its manifest.
    """
    k = _clamp_k(k)
    qv = embed_query(query)
    db = open_index()
    try:
        if not _has_transcript_table(db):
            return []
        # Over-fetch boundedly before metadata filtering. sqlite-vec's k is
        # applied at the virtual table; filtering joined metadata afterward
        # would otherwise risk returning too few results.
        fetch_k = min(50, max(k, k * 5))
        rows = db.execute(
            """
            SELECT d.path,d.vault,d.conversation_id,d.surface,d.host,
                   c.ordinal,c.role,c.source_line,c.timestamp,c.text,v.distance
            FROM vec_transcript_chunks v
            JOIN transcript_chunks c ON c.id=v.rowid
            JOIN transcript_docs d ON d.id=c.doc_id
            WHERE v.embedding MATCH ? AND k=?
            ORDER BY v.distance
            """,
            (qv.tobytes(), fetch_k),
        ).fetchall()
    finally:
        db.close()

    results = []
    manifest_cache: dict[str, dict] = {}
    for (
        path, row_vault, conversation_id, row_surface, row_host,
        ordinal, role, source_line, timestamp, text, distance,
    ) in rows:
        if vault and row_vault != vault:
            continue
        if surface and row_surface != surface:
            continue
        if host and row_host != host:
            continue
        if path not in manifest_cache:
            manifest_cache[path] = _history_manifest(path)
        manifest = manifest_cache[path]
        results.append({
            "evidence_path": path,
            "manifest_path": _history_manifest_path(path),
            "raw_path": manifest.get("raw_path", ""),
            "card_path": manifest.get("card_path", ""),
            "vault": row_vault,
            "conversation_id": conversation_id,
            "surface": row_surface,
            "host": row_host,
            "chunk_ordinal": int(ordinal),
            "role": role,
            "source_line": int(source_line or 0),
            "timestamp": timestamp or "",
            "score": round(1.0 - float(distance), 4),
            "excerpt": text.strip(),
        })
        if len(results) >= k:
            break
    return results


def read_history(
    evidence_path: str,
    start_chunk: int = 0,
    limit: int = 6,
    max_chars: int = 16000,
) -> dict:
    """Read bounded neighboring normalized chunks from one indexed conversation."""
    if not isinstance(evidence_path, str) or not evidence_path:
        return {"ok": False, "error": "evidence_path must be a non-empty string"}
    if os.path.isabs(evidence_path) or ".." in evidence_path.replace("\\", "/").split("/"):
        return {"ok": False, "error": "invalid evidence_path"}
    start = _range_int(start_chunk, "start_chunk", 0)
    count = _range_int(limit, "limit", 1, 20)
    char_cap = _range_int(max_chars, "max_chars", 1, 65536)
    db = open_index()
    try:
        if not _has_transcript_table(db):
            return {"ok": False, "error": "transcript evidence index unavailable"}
        doc = db.execute(
            "SELECT id,vault,conversation_id,surface,host,chunk_count "
            "FROM transcript_docs WHERE path=?",
            (evidence_path,),
        ).fetchone()
        if doc is None:
            return {"ok": False, "error": "evidence_path not indexed"}
        rows = db.execute(
            "SELECT ordinal,role,source_line,timestamp,text FROM transcript_chunks "
            "WHERE doc_id=? AND ordinal>=? ORDER BY ordinal LIMIT ?",
            (doc[0], start, count),
        ).fetchall()
    finally:
        db.close()

    out = []
    used = 0
    for ordinal, role, source_line, timestamp, text in rows:
        remaining = char_cap - used
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        out.append({
            "chunk_ordinal": int(ordinal),
            "role": role,
            "source_line": int(source_line or 0),
            "timestamp": timestamp or "",
            "text": excerpt,
            "truncated": len(excerpt) < len(text),
        })
        used += len(excerpt)
        if len(excerpt) < len(text):
            break
    manifest = _history_manifest(evidence_path)
    return {
        "ok": True,
        "evidence_path": evidence_path,
        "manifest_path": _history_manifest_path(evidence_path),
        "raw_path": manifest.get("raw_path", ""),
        "card_path": manifest.get("card_path", ""),
        "vault": doc[1],
        "conversation_id": doc[2],
        "surface": doc[3],
        "host": doc[4],
        "total_chunks": int(doc[5] or 0),
        "start_chunk": start,
        "returned_chunks": len(out),
        "returned_chars": used,
        "next_chunk": out[-1]["chunk_ordinal"] + 1 if out else start,
        "chunks": out,
    }


def _resolve_under(root: str, *parts: str) -> str:
    """Resolve components beneath root and reject traversal/symlink escapes."""
    root = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root, *parts))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError("path escapes containment root")
    return full


def _resolve(path: str) -> str:
    """Resolve a vault-relative path under ~/Vaults, guarding against escapes."""
    try:
        return _resolve_under(VAULTS_ROOT, path)
    except ValueError:
        raise ValueError(f"path escapes vaults root: {path}") from None


def _read_excerpt(path: str, n: int) -> str:
    try:
        full = _resolve(path)
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(n + 1)
        return text[:n].strip()
    except Exception as e:
        return f"[excerpt unavailable: {e}]"


def read_note(path: str) -> str:
    full = _resolve(path)
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


_RANGE_ALLOWED_SUFFIXES = frozenset((".md", ".json"))
_RANGE_UNITS = frozenset(("bytes", "lines"))
_RANGE_DEFAULT_LIMIT = 16384
_RANGE_MAX_BYTES = 65536
_RANGE_MAX_LINES = 1000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _range_int(value, name, minimum, maximum=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _hash_and_count_open_file(handle):
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    handle.seek(0)
    while True:
        chunk = handle.read(65536)
        if not chunk:
            break
        digest.update(chunk)
        newline_count += chunk.count(b"\n")
        last_byte = chunk[-1:]
    total_lines = newline_count + (1 if handle.tell() and last_byte != b"\n" else 0)
    return digest.hexdigest(), total_lines


def read_vault_range(
    path: str,
    offset: int = 0,
    limit: int = _RANGE_DEFAULT_LIMIT,
    unit: str = "bytes",
    max_bytes: int = _RANGE_MAX_BYTES,
    expected_sha256: str = "",
    expected_mtime_ns: int = 0,
) -> dict:
    """Read a deterministic bounded page from a Markdown note or JSON transcript.

    `offset` is zero-based in the selected unit. Byte pages are exact byte
    ranges decoded as UTF-8 with replacement metadata. Line pages contain only
    complete lines and stop before any line that would exceed max_bytes; callers
    can switch to byte mode for an oversized JSONL event. The SHA-256 and mtime
    returned by page one can be supplied on later pages to fail closed on drift.
    """
    if not isinstance(path, str) or not path:
        return {"ok": False, "error": "path must be a non-empty string"}
    normalized = path.replace("\\", "/")
    if os.path.isabs(path):
        return {"ok": False, "error": "absolute path rejected"}
    if ".." in normalized.split("/"):
        return {"ok": False, "error": "'..' component rejected"}
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in _RANGE_ALLOWED_SUFFIXES:
        return {"ok": False, "error": "only .md and .json text files are readable"}
    if unit not in _RANGE_UNITS:
        return {"ok": False, "error": "unit must be bytes or lines"}
    if expected_sha256 and (
        not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256)
    ):
        return {"ok": False, "error": "expected_sha256 must be 64 lowercase hex characters"}
    try:
        offset_value = _range_int(offset, "offset", 0)
        byte_cap = _range_int(max_bytes, "max_bytes", 1, _RANGE_MAX_BYTES)
        limit_cap = _RANGE_MAX_BYTES if unit == "bytes" else _RANGE_MAX_LINES
        limit_value = _range_int(limit, "limit", 1, limit_cap)
        expected_mtime = _range_int(expected_mtime_ns, "expected_mtime_ns", 0)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if unit == "bytes":
        limit_value = min(limit_value, byte_cap)

    try:
        full = _resolve(path)
        if not os.path.isfile(full):
            return {"ok": False, "error": "file not found"}
        with open(full, "rb") as handle:
            before = os.fstat(handle.fileno())
            digest, total_lines = _hash_and_count_open_file(handle)
            metadata = {
                "path": path,
                "unit": unit,
                "total_bytes": before.st_size,
                "total_lines": total_lines,
                "sha256": digest,
                "mtime_ns": before.st_mtime_ns,
            }
            if expected_sha256 and digest != expected_sha256:
                return {"ok": False, "error": "content changed", **metadata}
            if expected_mtime and before.st_mtime_ns != expected_mtime:
                return {"ok": False, "error": "content changed", **metadata}

            decode_replacements = False
            blocked_by_oversize_line = False
            if unit == "bytes":
                start = min(offset_value, before.st_size)
                handle.seek(start)
                raw = handle.read(limit_value)
                end = start + len(raw)
                content = raw.decode("utf-8", errors="replace")
                decode_replacements = "\ufffd" in content
                returned_units = len(raw)
                next_offset = end
                eof = end >= before.st_size
            else:
                start = min(offset_value, total_lines)
                handle.seek(0)
                raw_parts = []
                accumulated_bytes = 0
                returned_units = 0
                for line_number, raw_line in enumerate(handle):
                    if line_number < start:
                        continue
                    if returned_units >= limit_value:
                        break
                    if accumulated_bytes + len(raw_line) > byte_cap:
                        blocked_by_oversize_line = returned_units == 0
                        break
                    raw_parts.append(raw_line)
                    accumulated_bytes += len(raw_line)
                    returned_units += 1
                raw = b"".join(raw_parts)
                content = raw.decode("utf-8", errors="replace")
                decode_replacements = "\ufffd" in content
                next_offset = start + returned_units
                eof = next_offset >= total_lines
                end = next_offset

            after = os.fstat(handle.fileno())
            if (
                before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                return {"ok": False, "error": "content changed during read", **metadata}
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        **metadata,
        "requested_offset": offset_value,
        "requested_limit": limit_value,
        "max_bytes": byte_cap,
        "returned_range": [start, end],
        "returned_units": returned_units,
        "returned_bytes": len(raw),
        "next_offset": next_offset,
        "eof": eof,
        "truncated": not eof,
        "decode_replacements": decode_replacements,
        "blocked_by_oversize_line": blocked_by_oversize_line,
        "content": content,
    }


# homelab-vault/to-<seat>/* and homelab-vault/from-<seat>/* are relay
# machinery, not ordinary notes — raw write_note is blocked there so nothing
# bypasses stage_prompt's validation (seat/lane/token checks) or clobbers a
# script-owned response lane. The retired library vault left the mesh
# in §B6, so only the live homelab-vault relay tree remains guarded here.
_RELAY_VAULT = "homelab-vault"  # live relay vault — stage_prompt writes here
_RELAY_VAULT_GUARDED = frozenset({"homelab-vault"})


def _relay_operational_guard(path: str) -> dict | None:
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[0] not in _RELAY_VAULT_GUARDED:
        return None
    seat_dir = parts[1]
    if seat_dir.startswith("to-"):
        return {
            "ok": False,
            "error": (
                f"raw write_note is blocked for {parts[0]}/{seat_dir}/* "
                "(relay staging lane) — use stage_prompt(seat, lane, token, "
                "content) to stage an outbound prompt"
            ),
        }
    if seat_dir.startswith("from-"):
        return {
            "ok": False,
            "error": (
                f"raw write_note is blocked for {parts[0]}/{seat_dir}/* "
                "(relay response lane) — inbound responses are written by the "
                "script-owned delivery path, not raw write_note"
            ),
        }
    return None




def _journal_inbox_guard(path: str) -> dict | None:
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[1:3] == ["journal", "inbox"]:
        return {
            "ok": False,
            "error": (
                "raw write_note is blocked for journal inbox files — use "
                "append_journal_entry(vault, kind, seat, stamp, text)"
            ),
        }
    return None


def _atomic_overwrite_text(path: str, content: str) -> None:
    """Durably replace a text file without exposing a truncated destination."""
    parent = os.path.dirname(path)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".write-note-", suffix=".tmp", dir=parent)
        mode = stat.S_IMODE(os.stat(path).st_mode) if os.path.exists(path) else 0o664
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
        dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


def write_note(path: str, content: str, mode: str = "overwrite") -> dict:
    """Write a .md note under ~/Vaults. Returns {ok, ...}.

    Guards: rejects absolute paths, '..' components, paths resolving outside
    VAULTS_ROOT, non-.md targets, and (see _relay_operational_guard) raw writes
    into homelab-vault's to-*/from-* relay lanes. mode in {overwrite,
    append, prepend}. Parent dirs are created only within the root.
    """
    if os.path.isabs(path):
        return {"ok": False, "error": f"absolute path rejected: {path}"}
    if ".." in path.replace("\\", "/").split("/"):
        return {"ok": False, "error": f"'..' component rejected: {path}"}
    guard = _relay_operational_guard(path)
    if guard is not None:
        return guard
    guard = _journal_inbox_guard(path)
    if guard is not None:
        return guard
    if mode not in ("overwrite", "append", "prepend"):
        return {"ok": False, "error": f"invalid mode: {mode}"}
    try:
        full = _resolve(path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not full.endswith(".md"):
        return {"ok": False, "error": f"not a .md file: {path}"}
    try:
        parent = os.path.dirname(full)
        # parent is guaranteed under root (full is); create only within root.
        os.makedirs(parent, exist_ok=True)
        if mode == "overwrite":
            _atomic_overwrite_text(full, content)
        elif mode == "append":
            with open(full, "a", encoding="utf-8") as f:
                f.write(content)
        else:  # prepend
            existing = ""
            if os.path.exists(full):
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    existing = f.read()
            with open(full, "w", encoding="utf-8") as f:
                f.write(content + existing)
        nbytes = len(content.encode("utf-8"))
        return {"ok": True, "path": path, "bytes": nbytes, "mode": mode}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


_JOURNAL_VAULT_RE = re.compile(r"^[a-z0-9][a-z0-9-]*-vault$")
_JOURNAL_KINDS = frozenset({"decisions", "learnings"})
_JOURNAL_SEATS = frozenset({
    "codex", "claude", "localworker",
    # nodes (WORKER1-1)
    "delta", "charlie", "alpha",
    # legacy seats, kept so historical journal paths stay recognised
    "worker1", "worker3", "worker2", "worker4",
})
_JOURNAL_STAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC / (\d{2}:\d{2}) (CST|CDT)$"
)
_JOURNAL_MAX_ENTRY_BYTES = 4096


def _journal_utc_date(stamp: str) -> str:
    match = _JOURNAL_STAMP_RE.fullmatch(stamp or "")
    if not match:
        raise ValueError("stamp must be YYYY-MM-DD HH:MM UTC / HH:MM CST|CDT")
    utc_dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    central = utc_dt.astimezone(ZoneInfo("America/Chicago"))
    expected = f"{utc_dt:%Y-%m-%d %H:%M} UTC / {central:%H:%M} {central.tzname()}"
    if stamp != expected:
        raise ValueError(f"stamp does not match America/Chicago conversion; expected {expected}")
    return utc_dt.strftime("%Y-%m-%d")


def append_journal_entry(vault: str, kind: str, seat: str, stamp: str, text: str) -> dict:
    """Append one validated, newline-terminated bullet to a daily inbox.

    The destination is derived, never caller-supplied:
    <vault>/journal/inbox/<decisions|learnings>/<seat>-<UTC-date>.md.
    """
    if not isinstance(vault, str) or not _JOURNAL_VAULT_RE.fullmatch(vault):
        return {"ok": False, "error": "invalid vault name"}
    if kind not in _JOURNAL_KINDS:
        return {"ok": False, "error": "kind must be decisions or learnings"}
    if seat not in _JOURNAL_SEATS:
        return {"ok": False, "error": "invalid journal seat"}
    if not isinstance(text, str):
        return {"ok": False, "error": "text must be a string"}
    text = text.strip()
    if not text:
        return {"ok": False, "error": "text must be non-empty"}
    if "\n" in text or "\r" in text:
        return {"ok": False, "error": "text must be a single line"}
    try:
        utc_date = _journal_utc_date(stamp)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}

    try:
        vault_root = _resolve(vault)
    except ValueError:
        return {"ok": False, "error": "vault path escapes vaults root"}
    if not os.path.isdir(vault_root):
        return {"ok": False, "error": "vault does not exist"}

    entry = f"- {stamp} — {text}\n"
    entry_bytes = entry.encode("utf-8")
    if len(entry_bytes) > _JOURNAL_MAX_ENTRY_BYTES:
        return {"ok": False, "error": "journal entry exceeds 4096 bytes"}

    rel_path = os.path.join(vault, "journal", "inbox", kind, f"{seat}-{utc_date}.md")
    try:
        full_path = _resolve_under(
            vault_root, "journal", "inbox", kind, f"{seat}-{utc_date}.md"
        )
    except ValueError:
        return {"ok": False, "error": "journal path escapes vault"}
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fd = os.open(full_path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            prefix = b""
            size = os.fstat(fd).st_size
            if size:
                os.lseek(fd, -1, os.SEEK_END)
                if os.read(fd, 1) != b"\n":
                    prefix = b"\n"
            payload = prefix + entry_bytes
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short journal append")
                view = view[written:]
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "path": rel_path,
        "bytes": len(entry_bytes),
        "kind": kind,
        "seat": seat,
        "stamp": stamp,
    }


# Relay/ledger machinery — never deletable via MCP. Two families of trees
# hold live seat-to-seat traffic: homelab-vault/to-<seat>|from-<seat>
# (current relay) and loupe-vault/to-<seat>|from-<seat> (preserved legacy
# lanes from before the deploy role moved to delta/Worker1). Guard by matching
# path components against the CURRENT seat roster rather than a finite list
# of directory names, so the check doesn't silently miss a seat.
_RELAY_VAULTS = frozenset({"homelab-vault", "loupe-vault"})
# Relay lanes are protected from deletion. Node lanes were added by WORKER1-1; the
# legacy seat lanes stay listed because their historical run records remain.
_RELAY_SEATS = frozenset({
    "delta", "charlie", "alpha", "localworker",
    "worker1", "worker3", "worker2", "worker4",
})
_PROTECTED_BASENAMES = frozenset(
    {
        "DECISIONS.md",
        "LEARNINGS.md",
        "prompts.md",
        "responses.md",
        "latest.md",
        "latest_response.md",
        "recon.md",
    }
)


def _is_relay_seat_dir(name: str) -> bool:
    """True if `name` is a to-<seat>/from-<seat> relay lane dir for a current seat."""
    for prefix in ("to-", "from-"):
        if name.startswith(prefix) and name[len(prefix):] in _RELAY_SEATS:
            return True
    return False


def _in_relay_tree(parts: list) -> bool:
    """True if a vault-relative path (already split into components) falls
    anywhere under a to-<seat>/from-<seat> relay lane inside a relay vault.
    Checks every component after the vault name (not just the first), so
    files nested arbitrarily deep inside a lane are still caught."""
    if not parts or parts[0] not in _RELAY_VAULTS:
        return False
    return any(_is_relay_seat_dir(seg) for seg in parts[1:])


def _in_journal_tree(parts: list) -> bool:
    """True if a vault-relative path falls under any <vault>/journal/{decisions,learnings}/
    — the sharded ledger shards, vault-agnostic (loupe-vault, homelab-vault, future vaults
    alike). Basename matching alone (_PROTECTED_BASENAMES) can't cover these since each shard
    is named by month (e.g. 2026-07.md), not a fixed filename."""
    return (
        len(parts) >= 3
        and parts[1] == "journal"
        and parts[2] in ("decisions", "learnings")
    )


def delete_note(path: str) -> dict:
    """Soft-delete a .md note: move it into ~/Vaults/.trash/<ts>/. Never rm.

    Returns {ok:true, original_path, trashed_to, deleted_at} on success, else
    {ok:false, error}. The delete time is encoded in the <ts> dir name (UTC
    YYYYMMDDTHHMMSSZ) — NOT mtime, which Syncthing rewrites.
    """
    if os.path.isabs(path):
        return {"ok": False, "error": f"absolute path rejected: {path}"}
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        return {"ok": False, "error": f"'..' component rejected: {path}"}
    try:
        full = _resolve(path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not full.endswith(".md"):
        return {"ok": False, "error": f"not a .md file: {path}"}

    # PROTECTED: relay lanes (anywhere in path), .trash itself, ledger basenames,
    # sharded ledger journal tree.
    if _in_relay_tree(parts):
        return {"ok": False, "error": "protected path"}
    if parts and parts[0] == ".trash":
        return {"ok": False, "error": "protected path"}
    if os.path.basename(full) in _PROTECTED_BASENAMES:
        return {"ok": False, "error": "protected path"}
    if _in_journal_tree(parts):
        return {"ok": False, "error": "protected path"}

    if not os.path.isfile(full):
        return {"ok": False, "error": f"not an existing file: {path}"}

    # relative path under the root, preserving subdirs
    rel = os.path.relpath(full, VAULTS_ROOT)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(VAULTS_ROOT, ".trash", ts, rel)
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(full, dest)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {
        "ok": True,
        "original_path": rel,
        "trashed_to": os.path.join(".trash", ts, rel),
        "deleted_at": ts,
    }


# --- stage_prompt: compatibility staging for the live homelab-vault relay ---
# prompt (homelab-vault/to-<seat>/<lane>/latest.md). Raw write_note is
# blocked for that destination (see _relay_operational_guard above) so every
# staged prompt goes through seat/lane/token validation here.
STAGE_VALID_SEATS = frozenset({
    "delta", "charlie", "alpha", "localworker",
    "worker1", "worker3", "worker2", "worker4",
})
STAGE_VALID_LANES = frozenset({"prompts", "recon"})

# Token families in current use: legacy `FLEET-BUILD-YYYYMMDD-slug` and the
# current per-seat `FLEET-<SEAT>-BUILD-YYYYMMDD-slug` (BUILD or RECON), e.g.
# FLEET-WORKER2-BUILD-20260710-safe-stage-prompt. One optional UPPER segment
# between FLEET- and BUILD|RECON- covers both; the slug must not be empty or
# start/end with a hyphen.
# WORKER1-5: new tokens carry no seat (FLEET-<LANE>-<date>-<slug>); the historical
# FLEET- form stays accepted so existing run records keep resolving. Kept
# byte-identical in intent to kicker._LAUNCH_TOKEN_RE.
_STAGE_TOKEN_RE = re.compile(
    r"^(?:"
    r"FLEET-(?:BUILD|RECON)-\d{8}-[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"|FLEET-(?:[A-Z0-9]+-)?(?:BUILD|RECON)-\d{8}-[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r")$"
)


def stage_prompt(
    seat: str, lane: str, token: str, content: str, archive_previous: bool = True
) -> dict:
    """Stage a relay prompt at homelab-vault/to-<seat>/<lane>/latest.md.

    The only validated way to write that destination — see module docstring
    above _relay_operational_guard for why raw write_note is blocked there.

    Validation (in order): seat must be one of STAGE_VALID_SEATS; lane must be
    one of STAGE_VALID_LANES; content must be a non-empty string; token must be
    a non-empty string matching _STAGE_TOKEN_RE; token must occur verbatim in
    content. Never echoes content in an error.

    Write behavior: ensures the parent dir exists under ~/Vaults, writes
    through a temp file in the same directory then os.replace (atomic). If
    archive_previous and latest.md already exists, the previous content is
    copied to homelab-vault/to-<seat>/<lane>/archive/<UTC ts>-latest.md
    before the replacement. Staging always fully replaces latest.md — never
    appends/prepends.

    Returns {ok:True, path, seat, lane, token, bytes, archive_path?} on
    success, else {ok:False, error}.
    """
    if seat not in STAGE_VALID_SEATS:
        return {"ok": False, "error": f"invalid seat: {seat}"}
    if lane not in STAGE_VALID_LANES:
        return {"ok": False, "error": f"invalid lane: {lane}"}
    if not isinstance(content, str) or not content.strip():
        return {"ok": False, "error": "content must be a non-empty string"}
    if not isinstance(token, str) or not token.strip():
        return {"ok": False, "error": "token must be a non-empty string"}
    token = token.strip()
    if not _STAGE_TOKEN_RE.match(token):
        return {"ok": False, "error": "malformed token"}
    if token not in content:
        return {"ok": False, "error": "token not present in content"}

    rel_dir = os.path.join(_RELAY_VAULT, f"to-{seat}", lane)
    try:
        full_dir = _resolve(rel_dir)
    except ValueError:
        return {"ok": False, "error": "path escapes vaults root"}

    full_path = os.path.join(full_dir, "latest.md")
    rel_path = os.path.join(rel_dir, "latest.md")

    try:
        os.makedirs(full_dir, exist_ok=True)

        archive_rel = None
        if archive_previous and os.path.isfile(full_path):
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_dir = os.path.join(full_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            archive_full = os.path.join(archive_dir, f"{ts}-latest.md")
            shutil.copy2(full_path, archive_full)
            archive_rel = os.path.join(rel_dir, "archive", f"{ts}-latest.md")

        tmp_path = os.path.join(full_dir, f".latest.md.tmp{os.getpid()}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, full_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

        result = {
            "ok": True,
            "path": rel_path,
            "seat": seat,
            "lane": lane,
            "token": token,
            "bytes": len(content.encode("utf-8")),
        }
        if archive_rel is not None:
            result["archive_path"] = archive_rel
        return result
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

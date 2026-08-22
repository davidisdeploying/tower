#!/usr/bin/env python3
"""Create verified, point-in-time backups of Alpha SQLite state.

Covers both control-plane databases and the application databases this host
gained during the 2026-08-06/07 consolidation. Delta is the off-host
destination (see control_plane_offhost_sync.py)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone


SOURCES = {
    # Control-plane state.
    "nexus-events": Path.home() / ".local/state/nexus/events.db",
    "nexus-jobs-history": Path.home() / ".local/state/nexus/jobs_history.db",
    "model-usage-history": Path.home()
    / ".local/share/nexus/model-usage-history.sqlite3",
    "tower-quota-router": Path.home() / ".local/state/fleet/quota-router.sqlite3",
    "mediawatch": Path.home() / ".local/state/mediawatch/mediawatch.db",
    # Application state, added 2026-08-07. Prospect and Waypoint arrived from
    # delta on 2026-08-06 and Study Library from charlie on 2026-08-07; none of
    # them had any off-host copy afterwards. Their own backup timers, where they
    # exist at all, write to the same SD card as the live database. Absent
    # sources are tolerated and recorded in missing_optional_sources, so this
    # stays correct if an app ever moves off this host again.
    "prospect": Path.home() / "prospect/data/prospect.db",
    "waypoint": Path.home() / "waypoint/data/waypoint.db",
    "study-library": Path.home() / "waypoint/study-library/data/study_library.db",
    # Compendium workspace. Also protected by compendium-workspace-backup
    # (encrypted, off-host to charlie); captured here too so one mechanism covers
    # every live database in the fleet.
    "compendium-workspace": Path.home() / "compendium-data/workspace.sqlite3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_backup(source: Path, destination: Path) -> None:
    source_uri = f"file:{source}?mode=ro"
    with sqlite3.connect(source_uri, uri=True, timeout=30) as source_db:
        with sqlite3.connect(destination) as destination_db:
            source_db.backup(destination_db)
            result = destination_db.execute("PRAGMA quick_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError(f"quick_check failed for {source}: {result!r}")


def prune(root: Path, keep_days: int, now_epoch: float) -> list[str]:
    removed: list[str] = []
    threshold = now_epoch - keep_days * 86400
    for candidate in sorted(root.glob("20????????T??????Z")):
        if candidate.is_dir() and candidate.stat().st_mtime < threshold:
            shutil.rmtree(candidate)
            removed.append(candidate.name)
    return removed


def run(root: Path, keep_days: int) -> dict:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)

    final_dir = root / stamp
    if final_dir.exists():
        raise FileExistsError(final_dir)

    with tempfile.TemporaryDirectory(prefix=f".{stamp}.", dir=root) as temp_name:
        temp_dir = Path(temp_name)
        records: list[dict] = []
        missing: list[str] = []
        for label, source in SOURCES.items():
            if not source.is_file():
                missing.append(str(source))
                continue
            destination = temp_dir / f"{label}.sqlite3"
            sqlite_backup(source, destination)
            os.chmod(destination, 0o600)
            records.append(
                {
                    "label": label,
                    "source": str(source),
                    "backup": destination.name,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                    "quick_check": "ok",
                }
            )

        if not records:
            raise RuntimeError("no configured SQLite sources were present")

        manifest = {
            "schema": "tower-control-plane-backup-v1",
            "created_at": now.isoformat(),
            "host": os.uname().nodename,
            "retention_days": keep_days,
            "records": records,
            "missing_optional_sources": missing,
        }
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        os.chmod(manifest_path, 0o600)
        temp_dir.rename(final_dir)

    latest = root / "latest"
    temporary_link = root / f".latest.{stamp}"
    temporary_link.symlink_to(final_dir.name)
    temporary_link.replace(latest)
    removed = prune(root, keep_days, now.timestamp())
    return {"snapshot": str(final_dir), "records": records, "removed": removed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "FleetDatabaseBackups/alpha-snapshots",
    )
    parser.add_argument("--keep-days", type=int, default=14)
    args = parser.parse_args()
    if args.keep_days < 2:
        parser.error("--keep-days must be at least 2")
    result = run(args.root.expanduser(), args.keep_days)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

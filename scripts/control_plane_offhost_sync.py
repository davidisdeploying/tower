#!/usr/bin/env python3
"""Atomically append this host's latest database snapshot to the backup node.

Runs on every host that produces a snapshot set (alpha: control-plane state;
charlie: Loupe derived databases). The defaults are derived from the running
host so one canonical copy is correct everywhere -- see default_set_name().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import subprocess
import sys


REMOTE_BEGIN = r"""
import shutil,time,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
stamp=sys.argv[2]
manifest_sha=sys.argv[3]
token=sys.argv[4]
root.mkdir(parents=True,exist_ok=True,mode=0o700)
final=root/stamp
if final.exists():
 manifest=final/"manifest.json"
 digest=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else ""
 if digest != manifest_sha:
  raise RuntimeError(f"existing snapshot mismatch: {final}")
 print(json.dumps({"state":"already_present","snapshot":str(final)}))
else:
 incoming=root/".incoming"/token
 # Sweep staging left behind by an earlier run that transferred but failed to
 # finalize. Those directories hold a full payload and nothing else removes
 # them, so without this a run of failures silently fills the backup node.
 staging=root/".incoming"
 if staging.is_dir():
  cutoff=time.time()-6*3600
  for stale in staging.iterdir():
   if stale.is_dir() and stale.name!=token and stale.stat().st_mtime<cutoff:
    shutil.rmtree(stale,ignore_errors=True)
    print(json.dumps({"state":"swept_stale_staging","path":str(stale)}))
 incoming.mkdir(parents=True,exist_ok=False,mode=0o700)
 print(json.dumps({"state":"transfer","incoming":str(incoming)}))
"""

REMOTE_FINALIZE = r"""
import hashlib,json,os,pathlib,sqlite3,sys
root=pathlib.Path(sys.argv[1])
stamp=sys.argv[2]
token=sys.argv[3]
manifest_sha=sys.argv[4]
incoming=root/".incoming"/token
final=root/stamp
manifest_path=incoming/"manifest.json"
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha:
 raise RuntimeError("manifest hash changed during transfer")
manifest=json.loads(manifest_path.read_text())
for record in manifest["records"]:
 name=record["backup"]
 if pathlib.PurePosixPath(name).name != name:
  raise RuntimeError(f"unsafe backup name: {name}")
 path=incoming/name
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest != record["sha256"]:
  raise RuntimeError(f"hash mismatch: {name}")
 if name.endswith(".sqlite3"):
  uri=f"file:{path}?mode=ro"
  with sqlite3.connect(uri,uri=True,timeout=30) as db:
   result=db.execute("PRAGMA quick_check").fetchone()
   if not result or result[0] != "ok":
    raise RuntimeError(f"quick_check failed: {name}")
 elif not record.get("quick_check"):
  raise RuntimeError(f"non-sqlite record carries no source-side check: {name}")
if final.exists():
 raise RuntimeError(f"refusing to replace existing snapshot: {final}")
incoming.rename(final)
receipt={
 "schema":"fleet-offhost-transfer-v1",
 "source_host":sys.argv[5] if len(sys.argv)>5 else "unknown",
 "destination_host":os.uname().nodename,
 "snapshot":stamp,
 "manifest_sha256":manifest_sha,
 "verification":"sha256; sqlite quick_check for .sqlite3; source-side check otherwise",
}
(final/"offhost-transfer-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
temporary=root/f".latest.{token}"
temporary.symlink_to(stamp)
temporary.replace(root/"latest")
print(json.dumps({"state":"transferred","snapshot":str(final),"receipt":receipt}))
"""


def default_set_name() -> str:
    """Snapshot set for the running host.

    Both defaults derive from this so a single canonical copy is correct on
    every host. Previously each box carried a hand-edited copy, and charlie's
    --target-root still read "alpha-snapshots"; a run without explicit flags
    would have pushed Loupe's snapshots into alpha's set on the backup node.
    """
    return f"{os.uname().nodename}-snapshots"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_run(argv: list[str], *, input_text: str | None = None) -> str:
    proc = subprocess.run(
        argv,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if proc.returncode:
        detail = " ".join(proc.stderr.split())[-600:]
        raise RuntimeError(f"command failed rc={proc.returncode}: {detail}")
    return proc.stdout.strip()


def validate_snapshot(root: Path) -> tuple[Path, str]:
    root = root.expanduser().resolve()
    latest = root / "latest"
    if not latest.is_symlink():
        raise RuntimeError(f"latest is not a symlink: {latest}")
    snapshot = latest.resolve(strict=True)
    if snapshot.parent != root or not snapshot.is_dir():
        raise RuntimeError(f"latest escapes snapshot root: {snapshot}")
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("manifest has no records")
    for record in records:
        name = record.get("backup")
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise RuntimeError(f"unsafe backup name: {name!r}")
        path = snapshot / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"source verification failed: {name}")
    return snapshot, sha256(manifest_path)


def sync(source_root: Path, target_host: str, target_root: str) -> dict:
    snapshot, manifest_sha = validate_snapshot(source_root)
    stamp = snapshot.name
    if not stamp.startswith("20") or len(stamp) != 16:
        raise RuntimeError(f"unexpected snapshot stamp: {stamp}")
    token = f"{stamp}-{secrets.token_hex(6)}"
    ssh_base = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        target_host,
        "--",
        "python3",
        "-",
    ]
    begin = json.loads(
        checked_run(
            [*ssh_base, target_root, stamp, manifest_sha, token],
            input_text=REMOTE_BEGIN,
        )
    )
    if begin["state"] == "already_present":
        return begin
    incoming = f"{target_root}/.incoming/{token}/"
    checked_run(
        [
            "rsync",
            "-a",
            "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
            f"{snapshot}/",
            f"{target_host}:{incoming}",
        ]
    )
    return json.loads(
        checked_run(
            [*ssh_base, target_root, stamp, token, manifest_sha, os.uname().nodename],
            input_text=REMOTE_FINALIZE,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    snapshot_set = default_set_name()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "FleetDatabaseBackups" / snapshot_set,
    )
    parser.add_argument("--target-host", default="delta")
    parser.add_argument(
        "--target-root",
        default=f"/home/david/FleetDatabaseBackups/{snapshot_set}",
    )
    args = parser.parse_args()
    result = sync(args.source_root, args.target_host, args.target_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

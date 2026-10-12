# Tower — source layout

## Active entrypoints
- `server.py` — the MCP server. Run by `tower.service` (systemd --user unit) as
  `.venv/bin/python server.py`, `WorkingDirectory=~/tower`. This is the
  live process serving `/mcp`.
- `kicker.py` — relay orchestration: staging/kicking prompts to seats, run-dir tracking,
  stale-run classification, relay audit.
- `vaultsearch.py` — vault read/write/search layer (semantic search, note read/write/delete,
  stage_prompt, relay-lane guards) used by `server.py`.

## Tests
Run individually with `python -m unittest <module> -v` from this directory (venv active).
All are self-contained (tempdir/monkeypatch fixtures) — no live vault, network, NAS, or /proc.
- `test_cf_audience_config.py` — `server._load_cf_aud` and the CF-Access fail-closed gate.
- `test_delete_relay_guard.py` — `vaultsearch.delete_note`'s protection of relay machinery
  (to-*/from-* lanes) across all seats, including legacy loupe-vault lanes.
- `test_migration_charlie_compendium.py` — guards against regressions that repoint relay
  paths back to the retired loupe-vault relay root.
- `test_relay_audit.py` — `kicker.relay_audit`.
- `test_stage_prompt.py` — `vaultsearch.stage_prompt` and the `write_note` raw-write guard.
- `test_stale_status.py` — stale-run classification in `kicker.py`.

## Required runtime directories
- `.venv/` — the service's Python virtualenv (`ExecStart` interpreter).
- `model/` — local model artifacts.
- `~/.local/share/tower/index/` — receive-only Syncthing vault search index; runtime data, not source.
- `backups/<token>/` — active rollback bundles from recent audited changes. Each build that
  touches live source should write its pre-change backup here under a token-named
  subdirectory.

## External archive
One-off/scratch scripts and superseded root-level `*.bak*` files that are no longer needed
beside live source, but not worth deleting outright, live outside the source tree at:

    ~/tower-source-archives/<UTC-timestamp>/

## Rule
Future rollback backups belong in `~/tower/backups/<token>/` (recent,
audited, tied to a specific change) or the external archive above (older/superseded) —
never as loose `*.bak*` files or one-off scripts sitting beside the live source files in
this directory.

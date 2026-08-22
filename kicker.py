"""Headless relay kicker: launch `claude -p` on a REMOTE seat from a staged
relay prompt and poll. This copy runs on worker2 (the Tower host), which
is NOT a Worker1 seat — BOTH seats are launched over SSH: Worker1 on delta,
Worker3 on charlie. Run-state for each seat lives in its vault-synced from-<seat>/runs
lane so it propagates back to worker2, where status() reads it (no SSH to poll).

Four MCP tools sit on top of this (server.py): summon_worker1/ask_worker1 -> kick_delta,
summon_worker3/ask_worker3 -> kick_charlie; status polls the synced run-state.

Design constraints (from recon FLEET-RECON-20260628-kick-cc, extended by the
worker2 port FLEET-BUILD-20260701-worker2-mcp):
  - FastMCP tools run on a sync worker thread, so a BLOCKING child would freeze
    the server (and the strategy seat's in-flight call). Launches are therefore
    fire-and-forget: write prompt.txt + run.sh into the seat's vault-synced runs
    dir, wait out the Syncthing race on the target host, `setsid nohup` THERE,
    and return at once.
  - Only the REMOTE run.sh writes status.json/done/stderr.log/transcript into
    the run dir; this side writes only prompt.txt and run.sh — so there is no
    Syncthing write-write conflict. status() reads liveness purely from the
    synced filesystem and never waits on the child.
  - Confirmed: `claude -p` ingests a long prompt from stdin (`... < prompt.txt`).
  - `--dangerously-skip-permissions` is SCOPED to the launched run only (a CLI
    flag on this one invocation); it is never persisted to ~/.claude/settings.json.
"""
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

from quota_router import QuotaRouter

HOME = os.path.expanduser("~")
TOWER_NODE = (os.getenv("TOWER_NODE")
              or os.getenv("FLEET_TOWER_NODE", "alpha")).strip().lower()

# Relay state (to-<seat>/from-<seat> lanes, runs, transcripts, sessions, audits)
# lives in the dedicated shared vault homelab-vault, NOT in the Loupe project
# vault (loupe-vault) — cut over 2026-07-10 (FLEET-WORKER2-BUILD-20260710-charlie-
# compendium-migration), then relocated 2026-07-20 from the drained charlie-
# compendium vault into homelab-vault (FLEET-WORKER2-BUILD-20260720-m1-relay-
# lane-move, §M1). TOWER_RELAY_ROOT env override enables instant rollback.
# Loupe project knowledge (DECISIONS.md, LEARNINGS.md, etc.) stays in
# loupe-vault; only relay machinery moved.
VAULT = (os.getenv("TOWER_RELAY_ROOT") or os.getenv("FLEET_RELAY_ROOT")
         or os.path.join(HOME, "Vaults", "homelab-vault"))

# Worker1 (delta) remote-launch config. The MCP now runs on alpha, so
# kick_delta SSHes delta and runs `claude -p` THERE, mirroring the Worker3 pattern
# below; the remote run.sh self-writes run-state into WORKER1_RUNS/<token>/ (Syncthing
# carries it back to alpha where status() reads it).
#   - CLAUDE_BIN_WORKER1: delta's claude lives under ~/.npm-global (a non-login ssh
#     won't have it on PATH).
#   - Worker seats are pinned to Sonnet 5 (policy 2026-07-09) — see WORKER1_MODEL.
# WORKER1-1 (2026-08-18): the fleet addresses nodes, not seats. New runs are written
# under from-<node>/; the legacy per-seat roots below stay readable so the ~600
# historical run records remain visible to status(), inspect_run() and
# relay_audit() under the forward-only rule.
DELTA_RUNS = os.path.join(VAULT, "from-delta", "runs")
DELTA_TRANSCRIPTS = os.path.join(VAULT, "from-delta", "transcripts")
CHARLIE_RUNS = os.path.join(VAULT, "from-charlie", "runs")
CHARLIE_TRANSCRIPTS = os.path.join(VAULT, "from-charlie", "transcripts")
ALPHA_RUNS = os.path.join(VAULT, "from-alpha", "runs")
ALPHA_TRANSCRIPTS = os.path.join(VAULT, "from-alpha", "transcripts")

# node -> the legacy seat whose records precede it
# WORKER1-2: nodes are being renamed alpha->alpha, delta->delta, charlie->charlie.
# Both names resolve during the migration; canonical_node() is the single place
# that decides which one wins, so the flip is one edit rather than many.
NODE_RENAME = {"alpha": "alpha", "delta": "delta", "charlie": "charlie"}
NODE_RENAME_REVERSE = {v: k for k, v in NODE_RENAME.items()}


def canonical_node(name: str) -> str:
    """Resolve either the old or the new node name to the new one.

    The hosts answer to alpha/delta/charlie as of 2026-08-18; the pre-rename
    names stay accepted so stored routes, older callers and historical run
    records keep resolving.
    """
    if not name:
        return name
    name = name.strip().lower()
    return NODE_RENAME.get(name, name)


LEGACY_SEAT_FOR_NODE = {"delta": "worker1", "charlie": "worker3", "alpha": "worker2"}
# Which node a seat's work runs on. localworker runs on charlie, so it maps there -
# use this for placement questions only.
NODE_FOR_LEGACY_SEAT = {"worker1": "delta", "worker3": "charlie", "worker2": "alpha",
                        "localworker": "charlie", "localworker": "charlie", "worker4": "macbook",
                        # pre-rename node names resolve too
                        "delta": "delta", "charlie": "charlie", "alpha": "alpha"}

# Which launcher handles a seat name. localworker and worker4 keep their own launchers:
# aliasing localworker to its node would route zero-quota local work to a cloud
# worker, which is the opposite of why that path exists.
LAUNCHER_FOR_LEGACY_SEAT = {"worker1": "delta", "worker3": "charlie", "worker2": "alpha",
                            "localworker": "localworker",
                            "delta": "delta", "charlie": "charlie", "alpha": "alpha"}

WORKER1_RUNS = os.path.join(VAULT, "from-worker1", "runs")
WORKER1_TRANSCRIPTS = os.path.join(VAULT, "from-worker1", "transcripts")
CLAUDE_BIN_WORKER1 = "/home/david/.npm-global/bin/claude"
WORKER1_MODEL = "sonnet"  # pinned Sonnet 5 — worker seats never default to Opus
WORKER1_HOST = "david@delta"
# Retired under DL-15 (bare aliases canonical) / SSH-1 2026-07-24 — no longer read by _kick_remote.

# Worker3 (charlie) remote-launch config. kick_charlie SSHes charlie and runs `claude -p`
# THERE; the remote run.sh self-writes run-state into WORKER3_RUNS/<token>/ (Syncthing
# carries it back — no SSH needed to poll).
#   - CLAUDE_BIN_WORKER3: charlie's claude is at a DIFFERENT path than delta's (recon
#     FLEET-RECON-20260628-summon-worker3) and a non-login ssh won't have it on PATH.
#   - WORKER3_HOST resolves via Tailscale MagicDNS; WORKER3_HOST_IP is the agentless fallback.
WORKER3_RUNS = os.path.join(VAULT, "from-worker3", "runs")
WORKER3_TRANSCRIPTS = os.path.join(VAULT, "from-worker3", "transcripts")
CLAUDE_BIN_WORKER3 = "/home/david/.local/bin/claude"
# Worker3 (charlie) relay launches are pinned to Sonnet 5 (policy 2026-07-09: worker
# seats are Sonnet-5-only, Opus is reserved for the strategy seat). One constant
# referenced by kick_charlie (covers BOTH summon_worker3 and ask_worker3); flip here.
WORKER3_MODEL = "sonnet"
WORKER3_HOST = "david@charlie"
# Retired under DL-15 (bare aliases canonical) / SSH-1 2026-07-24 — no longer read by _kick_remote.
# Localworker was named Localworker until 2026-08-18. Its 197 historical run records
# live under from-localworker/, so the data path stays there under the
# forward-only rule; WORKER1-1 restructures run roots to from-<node>/.
LOCALWORKER_RUNS = os.path.join(VAULT, "from-localworker", "runs")
LOCALWORKER_TRANSCRIPTS = os.path.join(VAULT, "from-localworker", "transcripts")
CLAUDE_BIN_LOCALWORKER = CLAUDE_BIN_WORKER3
LOCALWORKER_MODEL = "gpt-oss:20b"
LOCALWORKER_HOST = WORKER3_HOST
# Retired under DL-15 (bare aliases canonical) / SSH-1 2026-07-24 — no longer read by _kick_remote.
# LOCALWORKER_HOST_IP = WORKER3_HOST_IP
# Worker4 (MacBook) remote-launch config. Converted 2026-07-13 (FLEET-WORKER2-BUILD-
# 20260713-worker4-claudecode-seat) from a bespoke MLX-CLI launcher to a proper
# Claude Code seat — kick_worker4 now runs on the shared _kick_remote SSH path,
# mirroring kick_delta/kick_charlie. WORKER4_HOST resolves via Tailscale MagicDNS;
# WORKER4_HOST_IP is the agentless fallback.
# The Mac's OWN absolute path to the same Syncthing-synced vault — different
# home root (davidgomez vs. david) than alpha/delta/charlie, so run.sh and the
# ssh wait-loop (which execute ON the Mac) need this instead of VAULT. See
# _kick_remote's remote_vault parameter.
WORKER4_VAULT = "/Users/davidgomez/Vaults/homelab-vault"
WORKER4_RUNS = os.path.join(VAULT, "from-worker4", "runs")
WORKER4_TRANSCRIPTS = os.path.join(VAULT, "from-worker4", "transcripts")
CLAUDE_BIN_WORKER4 = "/opt/homebrew/bin/claude"
WORKER4_MODEL = "sonnet"  # pinned Sonnet 5 (policy 2026-07-09) — worker seats never default to Opus
WORKER4_HOST = "davidgomez@macbook"
# Retired under DL-15 (bare aliases canonical) / SSH-1 2026-07-24 — no longer read by _kick_remote.

# Worker2 (LOCAL) seat. Same box as the MCP -> no ssh, no self-loop.
WORKER2_RUNS = os.path.join(VAULT, "from-worker2", "runs")
WORKER2_TRANSCRIPTS = os.path.join(VAULT, "from-worker2", "transcripts")
CLAUDE_BIN_WORKER2 = "/home/david/.local/bin/claude"
WORKER2_MODEL = "sonnet"   # pinned Sonnet 5 (policy 2026-07-09) — worker seat, not strategy
WORKER2_HOST = "david@alpha"

# Provider-aware worker lane. Codex is isolated from the Remote Control strategy
# profile and pinned to the fleet's worker-equivalent model. Automatic selection
# stays backward compatible (Claude first) unless the latest Claude run ended in
# a recognizable quota/rate-limit failure and a fresh Codex quota snapshot says
# Codex remains available. Tower never silently re-fires a failed token.
CODEX_BIN = "/home/david/.local/bin/codex"
CODEX_HOME_WORKER = "/home/david/.codex-worker"
CODEX_WORKER_MODEL = "gpt-5.6-terra"
AGY_BIN = "/home/david/.local/bin/agy"
GEMINI_WORKER_MODEL = "gemini-3.6-flash-high"
PROVIDERS = ("auto", "claude", "codex", "gemini")
QUOTA_ROOT = os.path.join(VAULT, "heartbeats", "quota")
QUOTA_ROUTER_DB = os.path.join(
    HOME, ".local", "state", "fleet", "quota-router.sqlite3"
)
GEMINI_WORKER_MARKER = os.path.join(
    HOME, ".config", "fleet", "gemini-headless-worker-enabled"
)
_QUOTA_ERROR_RE = re.compile(
    r"(usage limit|rate limit|quota|capacity.*exhaust|resets? at)", re.IGNORECASE
)
_PROVIDER_FAILURE_RE = re.compile(
    r"(authentication[_ -]?failed|not logged in|unauthorized|invalid api key|"
    r"usage limit|rate limit|quota|capacity.*exhaust|resets? at|"
    r"provider unavailable|upstream.*(?:timeout|unavailable)|"
    r"(?:api|provider).*(?:connection|transport).*(?:failed|error))",
    re.IGNORECASE,
)

# Every provider receives the same artifact-placement invariant. The authoritative
# text lives in the synchronized fleet contract so one Mac-side AGENTS.md edit
# updates Claude, Codex, and Localworker launches without copying worker profiles.
# request.md remains the caller's immutable source; only executable prompt.txt
# receives this launcher-owned invariant.
_WORKER_ARTIFACT_BEGIN = "<!-- FLEET_WORKER_ARTIFACT_CONTRACT_V1_BEGIN -->"
_WORKER_ARTIFACT_END = "<!-- FLEET_WORKER_ARTIFACT_CONTRACT_V1_END -->"
_WORKER_ARTIFACT_FALLBACK = """## Worker artifact invariant

When a dispatched task produces a bounded durable non-Markdown project artifact,
write it only under the owning vault's files/YYYY/YYYY-MM-DD/<seat>-<task-slug>/
directory with a MANIFEST.md and SHA-256 values. Never use files/ for repositories,
dependencies, caches, runtimes, models, indexes, live databases, secrets,
uncontrolled dumps, or unbounded logs. If the owning vault or exact artifact
destination is missing, stop and report that routing gap rather than guessing."""

# Relay lanes: builds flow through "prompts", recon through "recon". Each lane has
# its own staged-prompt and return paths so a recon answer lands in the recon
# return lane and a build answer in the prompts return lane.
LANES = ("prompts", "recon")

# A relay-token-shaped string: UPPER head + >=2 dash-separated segments, e.g.
# FLEET-BUILD-20260628-kick-cc. Used only to surface a hint on token mismatch.
_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Za-z0-9]+){2,}\b")

# Strict relay-token grammar, intentionally identical to vaultsearch.py's
# _STAGE_TOKEN_RE.  stage_prompt validates the shared latest.md slot, but the
# launch boundary is independently reachable and must validate again before a
# token is used in a path or substituted into a shell/SSH script.
#
# WORKER1-5: seats are retired, so a new token carries no seat: FLEET-<LANE>-<date>-<slug>.
# The historical FLEET-<SEAT>-<LANE>-<date>-<slug> form stays accepted because
# ~35,000 run records in the vault are named that way and must keep resolving.
_LAUNCH_TOKEN_RE = re.compile(
    r"^(?:"
    r"FLEET-(?:BUILD|RECON)-\d{8}-[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"|FLEET-(?:[A-Z0-9]+-)?(?:BUILD|RECON)-\d{8}-[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r")$"
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _worker_artifact_contract() -> str:
    """Return the synchronized provider-neutral artifact invariant.

    The marked block is deliberately extracted at launch time, not import time,
    so a Syncthing-delivered AGENTS.md update applies to the next dispatch without
    restarting Tower. A bounded fallback preserves the safety boundary if the
    source file is temporarily unavailable or malformed.
    """
    path = os.path.join(
        VAULT, "conventions", "fleet-strategy-contract", "AGENTS.md"
    )
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        start = source.index(_WORKER_ARTIFACT_BEGIN) + len(_WORKER_ARTIFACT_BEGIN)
        end = source.index(_WORKER_ARTIFACT_END, start)
        contract = source[start:end].strip()
        if contract:
            return contract
    except (OSError, ValueError):
        pass
    return _WORKER_ARTIFACT_FALLBACK


def _append_worker_artifact_contract(prompt: str) -> str:
    return (
        prompt.rstrip()
        + "\n\n---\n\n"
        + _worker_artifact_contract().strip()
        + "\n"
    )


def _append_worker_execution_context(prompt: str, execution_context: str) -> str:
    """Append launcher-owned target/transport facts without changing request.md."""
    if not execution_context:
        return prompt
    return (
        prompt.rstrip()
        + "\n\n---\n\n"
        + "## ROUTED EXECUTION CONTEXT (launcher-owned)\n\n"
        + execution_context.strip()
        + "\n"
    )


def _codex_quota_available(seat: str) -> tuple[bool, str]:
    path = os.path.join(QUOTA_ROOT, f"{seat}-codex.json")
    try:
        with open(path, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        generated = datetime.fromisoformat(snapshot["generated_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - generated).total_seconds()
        if age > 900:
            return False, "codex_quota_stale"
        if not snapshot.get("ok"):
            return False, "codex_quota_unavailable"
        limits = snapshot.get("rateLimits") or {}
        if limits.get("rateLimitReachedType") or limits.get("spendControlReached"):
            return False, "codex_limit_reached"
        windows = [limits.get("primary"), limits.get("secondary")]
        if any((window or {}).get("usedPercent", 0) >= 100 for window in windows):
            return False, "codex_window_exhausted"
        return True, "codex_quota_available"
    except Exception:
        return False, "codex_quota_missing"


def _latest_claude_quota_failure(runs_root: str) -> bool:
    try:
        dirs = sorted(
            (
                entry for entry in os.scandir(runs_root)
                if entry.is_dir(follow_symlinks=False)
            ),
            key=lambda entry: entry.stat(follow_symlinks=False).st_mtime,
            reverse=True,
        )
    except OSError:
        return False
    for entry in dirs[:5]:
        try:
            with open(os.path.join(entry.path, "status.json"), encoding="utf-8") as handle:
                status_payload = json.load(handle)
            if status_payload.get("provider", "claude") != "claude":
                continue
            done_path = os.path.join(entry.path, "done")
            if not os.path.exists(done_path):
                return False
            with open(done_path, encoding="utf-8") as handle:
                if handle.read().strip() == "0":
                    return False
            with open(
                os.path.join(entry.path, "stderr.log"),
                encoding="utf-8",
                errors="replace",
            ) as handle:
                return bool(_QUOTA_ERROR_RE.search(handle.read()[-12000:]))
        except OSError:
            continue
        except (ValueError, TypeError):
            continue
    return False


def _quota_router() -> QuotaRouter:
    return QuotaRouter(QUOTA_ROOT, QUOTA_ROUTER_DB, GEMINI_WORKER_MARKER)


def _terminal_provider_failure(run_dir: str, terminal: dict | None) -> bool:
    """Classify only bounded provider/auth/quota evidence as provider failure."""
    evidence = ""
    stderr_path = os.path.join(run_dir, "stderr.log")
    try:
        with open(stderr_path, encoding="utf-8", errors="replace") as handle:
            evidence = handle.read()[-12000:]
    except OSError:
        pass
    if terminal:
        try:
            evidence += "\n" + json.dumps(terminal, sort_keys=True)[-4000:]
        except (TypeError, ValueError):
            pass
    return bool(_PROVIDER_FAILURE_RE.search(evidence))


def _provider_spec(
    seat: str,
    requested: str,
    runs_root: str,
    task_size: str = "small",
    reservation_id: str | None = None,
    max_runtime_seconds: int = 0,
) -> dict:
    requested = (requested or "auto").strip().lower()
    if requested not in PROVIDERS:
        return {"ok": False, "error": "bad provider"}
    reserve = bool(reservation_id)
    ttl_seconds = max(max_runtime_seconds or 21600, 60) + 900
    decision = _quota_router().recommend(
        lane="worker",
        size=task_size,
        allowed_providers=("claude", "codex", "gemini"),
        explicit_provider=None if requested == "auto" else requested,
        reserve=reserve,
        reservation_id=reservation_id,
        ttl_seconds=min(ttl_seconds, 604800),
    )
    if not decision.get("ok"):
        return decision
    provider = decision["provider"]
    if provider == "codex":
        binary, model = CODEX_BIN, CODEX_WORKER_MODEL
    elif provider == "gemini":
        binary, model = AGY_BIN, GEMINI_WORKER_MODEL
    else:
        binary, model = None, None
    return {
        "ok": True,
        "provider": provider,
        "bin": binary,
        "model": model,
        "reason": decision["reason"],
        "routing": decision,
        "reservation_id": reservation_id if reserve else None,
    }


def _entry_provider_spec(
    seat: str,
    requested: str,
    runs_root: str,
    token,
    lane: str,
    max_runtime_seconds: int,
    prompt_content,
    task_size: str = "small",
) -> dict:
    # Compatibility-only staged launches (legacy ask_*/summon_*) retain their
    # historical zero-extra-I/O Claude path. Provider-aware auto routing belongs
    # to atomic dispatch, where the exact prompt is supplied directly.
    if (requested or "auto").strip().lower() == "auto" and prompt_content is None:
        return {
            "ok": True, "provider": "claude", "bin": None, "model": None,
            "reason": "legacy_claude",
        }
    normalized = (token or "").strip()
    if not normalized:
        return {"ok": False, "error": "empty token"}
    if not _LAUNCH_TOKEN_RE.fullmatch(normalized):
        return {"ok": False, "error": "malformed token"}
    if lane not in LANES:
        return {"ok": False, "error": "bad lane"}
    try:
        _normalize_max_runtime(max_runtime_seconds)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return _provider_spec(
        seat,
        requested,
        runs_root,
        task_size=task_size,
        reservation_id=normalized if prompt_content is not None else None,
        max_runtime_seconds=max_runtime_seconds,
    )


def _finish_routed_launch(
    result: dict, spec: dict, runs_root: str, token: str
) -> dict:
    reservation_id = spec.get("reservation_id")
    routing = spec.get("routing")
    if not result.get("ok"):
        if reservation_id:
            _quota_router().release(reservation_id, "launch_failed")
        return result
    if isinstance(routing, dict):
        run_dir = os.path.join(runs_root, token)
        path = os.path.join(run_dir, "routing.json")
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(routing, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, path)
        except OSError:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        result["routing"] = routing
    return result


# --- Stale-run classification -----------------------------------------------
# A run with no `done` sentinel is either still going (heartbeat/status.json
# fresh) or dead-and-abandoned (host crash, ssh drop, kill -9 — nothing left to
# write `done`). These two env-configurable thresholds turn "no news" into an
# explicit "stale" status instead of "running"/"launching" forever. Parsed
# defensively per-call (not cached at import) so a bad or updated env value
# takes effect on the next status() poll without a process restart.
_DEFAULT_STALE_AFTER_SECONDS = 7200
_DEFAULT_LAUNCH_GRACE_SECONDS = 600


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError:
        return default
    return val if val > 0 else default


def _stale_after_seconds() -> int:
    return _env_positive_int("TOWER_STALE_AFTER_SECONDS", _DEFAULT_STALE_AFTER_SECONDS)


def _launch_grace_seconds() -> int:
    return _env_positive_int("TOWER_LAUNCH_GRACE_SECONDS", _DEFAULT_LAUNCH_GRACE_SECONDS)


def _mtime_or_none(path: str):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age_seconds(epoch: float) -> int:
    return max(0, int(time.time() - epoch))


def _liveness(ref_epoch, threshold_seconds: int, stale_reason: str, base_status: str) -> dict:
    """Classify a no-`done` run given its freshness reference mtime.

    `ref_epoch` is None when even the fallback path (e.g. the run dir itself)
    couldn't be stat'd — in that case we can't judge age, so just report
    `base_status` with no age fields. Otherwise age is compared to
    `threshold_seconds`; past it the status becomes "stale" with `stale_reason`,
    additive fields `age_seconds`/`last_seen_at` are always included when a
    reference mtime is available.
    """
    fields = {"status": base_status}
    if ref_epoch is None:
        return fields
    age = _age_seconds(ref_epoch)
    fields["age_seconds"] = age
    fields["last_seen_at"] = _iso_from_epoch(ref_epoch)
    if age > threshold_seconds:
        fields["status"] = "stale"
        fields["stale_reason"] = stale_reason
    return fields


def _first_token_like_line(text: str) -> str:
    """Surface the first token-like line (else first non-empty line) as a hint."""
    for line in text.splitlines():
        if _TOKEN_RE.search(line):
            return line.strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _resolve_session_note(session_note: str) -> str:
    """Normalize a session-note reference to an absolute path for the footer.

    Accepts:
    - an absolute path, returned unchanged.
    - a project-relative path, e.g. "sessions/2026-07-21-slug.md"
      -> VAULT/sessions/2026-07-21-slug.md.
    - the same path spelled Vaults-relative and prefixed with VAULT's own
      directory name, e.g. "homelab-vault/sessions/2026-07-21-slug.md" ->
      VAULT/sessions/2026-07-21-slug.md — the SAME target as the
      project-relative form above. The prefix is stripped before joining
      onto VAULT so it isn't doubled (previously this joined onto VAULT
      unstripped, producing VAULT/homelab-vault/sessions/...).
    - a path into a SIBLING vault, e.g. "prospect-vault/sessions/x.md" ->
      ~/Vaults/prospect-vault/sessions/x.md (H7). Only the relay root's own
      name was stripped before, so every other vault fell through to the
      join below and landed at VAULT/prospect-vault/sessions/x.md — a stray
      nested stub under the relay root, with the real note never appended.
      That was reported against the retired "library/" prefix and
      looked vault-specific; it was not. It bites every non-relay vault, and
      Prospect (whose session notes are prospect-vault/sessions/...) is the
      one it was actually losing. Resolution is gated on the directory really
      existing under the Vaults root, and the result is containment-checked,
      so this widens where a note may be written by exactly one level and
      never outside ~/Vaults.
    - a bare slug, e.g. "2026-07-21-slug" -> VAULT/sessions/2026-07-21-slug.md.

    VAULT defaults to ~/Vaults/homelab-vault (overridable via
    TOWER_RELAY_ROOT).
    """
    sn = session_note.strip()
    if not sn:
        return ""
    if os.path.isabs(sn):
        return sn
    if "/" in sn:
        prefix = f"{os.path.basename(VAULT)}/"
        if sn.startswith(prefix):
            return os.path.join(VAULT, sn[len(prefix):])
        # H7: a sibling vault under the same Vaults root resolves there, not under
        # the relay root. Gated on the directory actually existing so an ordinary
        # subdirectory name (e.g. "sessions/...") still joins onto VAULT as before.
        vaults_root = os.path.dirname(VAULT)
        first = sn.split("/", 1)[0]
        if first.endswith("-vault") and os.path.isdir(os.path.join(vaults_root, first)):
            candidate = os.path.join(vaults_root, sn)
            root_real = os.path.realpath(vaults_root)
            if os.path.realpath(candidate).startswith(root_real + os.sep):
                return candidate
        return os.path.join(VAULT, sn)
    base = sn if sn.endswith(".md") else f"{sn}.md"
    return os.path.join(VAULT, "sessions", base)


def _build_worker_footer(
    token: str,
    session_note: str,
    response_path: str,
    responses_log: str,
    token_response_path: str,
    *,
    kick_name: str,
    launch_location: str,
    owner_label: str,
    halt: bool,
) -> str:
    """Build the shared manual-delivery contract for Claude Code seats."""
    location = f" on {launch_location}" if launch_location else ""
    lines = [
        "",
        "",
        "---",
        "",
        f"## RELAY DELIVERY (appended by {kick_name} — do this as the FINAL step)",
        "",
        f"You were launched headless{location} by the strategy seat for token `{token}`. "
        "When the work above is complete:",
        "",
        f"1. OVERWRITE `{response_path}` with your concise response (what you did, key "
        "results, verification, anything open), beginning with a header line and "
        f"ending by echoing the token `{token}`.",
    ]
    if token_response_path:
        lines += [
            f"2. ALSO write that same response to `{token_response_path}` (the "
            "per-token return slot — safe from being overwritten by a different run).",
            f"3. APPEND that same response to `{responses_log}`, prefixed with a line "
            "of `---`.",
        ]
    else:
        lines += [
            f"2. APPEND that same response to `{responses_log}`, prefixed with a line "
            "of `---`.",
        ]
    note_path = _resolve_session_note(session_note)
    if note_path:
        step_n = 4 if token_response_path else 3
        lines += [
            f"{step_n}. LOG to the session note `{note_path}` (READ-BEFORE-WRITE). Append "
            f"ONLY your {owner_label}-owned sections — `## Execution log`, `## Blockers & fixes`, "
            f"and `[{owner_label}]`-prefixed lines under `## Learnings` — in Option-A dual-voice. "
            "NEVER overwrite or rewrite the strategy-owned sections (`## Goal / "
            "context`, `## Decisions & rationale`, `## Open / next`). Append-only; if "
            "the note does not exist, create it with the standard headers and leave "
            "the strategy sections as empty stubs.",
        ]
    if halt:
        lines += ["", f"Then echo the token `{token}` and HALT."]
    return "\n".join(lines)


def _build_footer(
    token: str,
    session_note: str,
    response_path: str,
    responses_log: str,
    token_response_path: str = "",
) -> str:
    """Worker1 manual-delivery footer; output is protected by golden tests."""
    return _build_worker_footer(
        token, session_note, response_path, responses_log, token_response_path,
        kick_name="kick_delta", launch_location="", owner_label="Worker1", halt=False,
    )


def _build_worker3_footer(
    token: str,
    session_note: str,
    response_path: str,
    responses_log: str,
    token_response_path: str = "",
) -> str:
    """Worker3 manual-delivery footer; output is protected by golden tests."""
    return _build_worker_footer(
        token, session_note, response_path, responses_log, token_response_path,
        kick_name="kick_charlie", launch_location="Worker3 (charlie)",
        owner_label="CC", halt=True,
    )


def _build_worker4_footer(
    token: str,
    session_note: str,
    response_path: str,
    responses_log: str,
    token_response_path: str = "",
) -> str:
    """Worker4 manual-delivery footer; output is protected by golden tests."""
    return _build_worker_footer(
        token, session_note, response_path, responses_log, token_response_path,
        kick_name="kick_worker4", launch_location="Worker4 (MacBook)",
        owner_label="CC", halt=True,
    )


def _build_localworker_footer(
    token: str,
    session_note: str,
    response_path: str,
    responses_log: str,
    token_response_path: str = "",
) -> str:
    """The delivery footer kick_localworker appends to the staged to-localworker prompt.

    Unlike the other seats' footers, Localworker runs under fleet_agent.py's
    finish_task harness (2026-07-20, FLEET-WORKER2-BUILD-20260720-localworker-
    delivery-footer), NOT a full Claude Code worker — it has no read_file/
    write_file-driven session-note editing discipline and no append mode on
    write_file. finish_task's own handler (fleet_agent.py) already writes
    response_path, the token_response_path run-dir copy, and appends to
    responses_log verbatim from one tool call (see FLEET-WORKER3-RECON-
    20260720-gptoss-smoke-maxturns: the generic manual-delivery footer below
    made a production gpt-oss:20b run burn 9 of 15 tool calls hand-reconstructing
    those same writes, balloon context to ~26.7K tokens, and never reach
    finish_task). So this footer tells Localworker to do none of that manual
    work and call finish_task once instead. token_response_path/response_path/
    responses_log are accepted for signature parity with the other _build_*_footer
    functions (footer_fn is called uniformly by _kick_remote) but are not named in
    the instructions — finish_task already knows its own destinations from the
    --response-path/--responses-log/--run-dir flags baked into run.sh.
    """
    lines = [
        "",
        "",
        "---",
        "",
        "## RELAY DELIVERY (appended by kick_localworker — do this as the FINAL step)",
        "",
        f"You were launched headless on Localworker (charlie) by the strategy seat for token "
        f"`{token}`. This is the finish_task harness, not a full Claude Code worker — "
        "durable delivery is automated, not manual. When the work above is complete:",
        "",
        "1. Do NOT manually write or overwrite response.md, latest_response.md, or "
        "responses.md, and do NOT attempt an append-by-rewrite (read-then-write-back) "
        "delivery of any of those files.",
        "2. Complete the assigned task using your real tool results (read_file, "
        "write_file, edit_file, grep_search, list_dir, run_command as needed).",
        "3. Call the `finish_task` tool exactly once, with a concise `summary` and the "
        f"full `response_markdown` report (ending by echoing the token `{token}`). The "
        "harness writes response_path, the per-token run-dir copy, and appends "
        "responses_log for you from that single call — do not duplicate its work.",
    ]
    note_path = _resolve_session_note(session_note)
    if note_path:
        lines += [
            f"4. A session note (`{note_path}`) was supplied, but this harness has no "
            "supported mechanism for editing it — do NOT manually read/rewrite it "
            "unless a Localworker-specific session-note tool has been explicitly "
            "documented to you. Leave it alone and let finish_task's response_markdown "
            "carry your report instead.",
        ]
    return "\n".join(lines)


def _build_localworker_compact_footer(
    token: str,
    session_note: str,
    response_path: str,
    responses_log: str,
    token_response_path: str = "",
) -> str:
    """Localworker delivery footer for FLEET_COMPACT_DELIVERY_V1 prompts."""
    lines = [
        "",
        "",
        "---",
        "",
        "## RELAY DELIVERY (compact contract — do this as the FINAL step)",
        "",
        f"You were launched headless on Localworker (charlie) by the strategy seat for token "
        f"`{token}`. The immutable request contains a compact-delivery contract. "
        "Follow its exact declared tool sequence:",
        "",
        "1. Do NOT call the legacy `finish_task` tool.",
        "2. Do NOT manually write response.md, latest_response.md, responses.md, "
        "compact_result.json, or any relay session note.",
        "3. Use only the declared native tools, in the declared order, and call "
        "`finish_task_compact` exactly once as the final tool.",
        "4. Supply only the bounded compact outcome fields. The harness derives "
        "evidence from executed tool results, writes and hashes compact_result.json, "
        f"renders the final response, and places `{token}` on its last line.",
    ]
    note_path = _resolve_session_note(session_note)
    if note_path:
        lines += [
            f"5. The supplied session note is `{note_path}`. Leave it alone; compact "
            "delivery does not authorize Localworker to edit continuity.",
        ]
    return "\n".join(lines)


def _build_worker2_footer(
    token: str,
    session_note: str,
    response_path: str,
    responses_log: str,
    token_response_path: str = "",
) -> str:
    """Worker2 manual-delivery footer; output is protected by golden tests."""
    return _build_worker_footer(
        token, session_note, response_path, responses_log, token_response_path,
        kick_name="kick_alpha", launch_location="worker2 (local)",
        owner_label="Worker2", halt=True,
    )


# Backgrounded inside run.sh (on the target host) right after status.json is
# written. Refreshes __RUNDIR__/heartbeat roughly every 30s via write-to-tmp +
# atomic `mv` while run.sh is alive; `trap ... EXIT` kills the loop the moment
# run.sh exits (normally or via the `done` write), so a live run always has a
# fresh heartbeat and a dead one stops producing them within ~30s. This lets
# status() distinguish "still going" from "died without writing done" even
# across hosts, where no /proc probe is possible (see _liveness).
_HEARTBEAT_SNIPPET = (
    '( while true; do\n'
    '    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUNDIR/heartbeat.tmp.$$"\n'
    '    mv -f "$RUNDIR/heartbeat.tmp.$$" "$RUNDIR/heartbeat"\n'
    '    sleep 30\n'
    '  done ) &\n'
    'HB_PID=$!\n'
    "trap 'kill \"$HB_PID\" 2>/dev/null' EXIT\n"
)


def _mark_launch_failed(run_dir: str, reason: str) -> None:
    """Write a terminal sentinel into run_dir for a launch that never started run.sh.

    Every done-presence consumer (Panel's three run-state classifiers, plus this
    module's own _run_entry/_remote_status) already treats `done` as terminal, so
    writing it here needs zero downstream changes — a launch-failed run stops
    reading as phantom-BUSY for the full ORPHAN_AFTER_S window. Write-once: a
    pre-existing `done` (a real terminal state, or a run that actually started)
    is left untouched. Must never raise — a marker-write failure must not mask
    the original launch failure already being returned to the caller.
    """
    try:
        if not os.path.isdir(run_dir):
            return
        done_path = os.path.join(run_dir, "done")
        if os.path.exists(done_path):
            return
        # 70 == EX_SOFTWARE, repurposed as the launch-never-started sentinel so
        # every done-presence consumer flips this run to terminal/not-running.
        with open(done_path, "w", encoding="utf-8") as f:
            f.write("70\n")
        token = os.path.basename(os.path.normpath(run_dir))
        line = f"{_now()}\treason={reason}\ttoken={token}\n"
        with open(os.path.join(run_dir, "launch-failed.txt"), "w", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


_MAX_RUNTIME_SECONDS = 7 * 24 * 60 * 60


def _normalize_max_runtime(value):
    """Validate an opt-in deadline; zero preserves unlimited legacy behavior."""
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        raise ValueError("max_runtime_seconds must be 0 or an integer from 1 to 604800")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("max_runtime_seconds must be 0 or an integer from 1 to 604800")
    if str(parsed) != str(value).strip() or parsed < 0 or parsed > _MAX_RUNTIME_SECONDS:
        raise ValueError("max_runtime_seconds must be 0 or an integer from 1 to 604800")
    return parsed


# Per-run helper used only when a caller opts into a deadline. It gives every
# host (including macOS Worker4, which has no GNU timeout) identical process-group
# termination and a durable, atomic terminal reason. The launched worker owns a
# new session so TERM/KILL applies to its entire child tree, not run.sh/Tower.
_DEADLINE_RUNNER = r"""#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys


def atomic_json(path, payload):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


parser = argparse.ArgumentParser()
parser.add_argument("--max-runtime", type=int, required=True)
parser.add_argument("--terminal", required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
if args.command and args.command[0] == "--":
    args.command = args.command[1:]
if args.max_runtime <= 0 or not args.command:
    raise SystemExit(64)

proc = None
try:
    proc = subprocess.Popen(args.command, start_new_session=True)
    try:
        raise SystemExit(proc.wait(timeout=args.max_runtime))
    except subprocess.TimeoutExpired:
        timed_out_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        escalated = False
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            escalated = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        atomic_json(args.terminal, {
            "reason": "max_runtime_exceeded",
            "max_runtime_seconds": args.max_runtime,
            "timed_out_at": timed_out_at,
            "worker_pid": proc.pid,
            "worker_returncode": proc.returncode,
            "kill_escalated": escalated,
        })
        raise SystemExit(124)
except SystemExit:
    raise
except Exception as exc:
    atomic_json(args.terminal, {
        "reason": "deadline_runner_error",
        "max_runtime_seconds": args.max_runtime,
        "error_type": type(exc).__name__,
    })
    raise SystemExit(70)
"""


def _kick_remote(
    token: str,
    lane: str,
    session_note: str,
    *,
    seat: str,
    to_seat: str,
    from_seat: str,
    runs_root: str,
    transcripts_dir: str,
    claude_bin: str,
    model,
    host: str,
    host_ip: str,
    seat_env: str,
    footer_fn,
    marker: str,
    launched_note: str,
    local: bool = False,
    remote_vault: str = None,
    prompt_content: str = None,
    max_runtime_seconds: int = 0,
    provider: str = "claude",
    provider_reason: str = "legacy_claude",
    execution_context: str = "",
    worker_routing: dict = None,
) -> dict:
    """Shared remote launcher used by kick_delta and kick_charlie.

    Writes prompt.txt + run.sh into <runs_root>/<token>/ (which Syncthing carries
    to the target host), then SSHes `host` and `setsid nohup bash run.sh` — the
    ssh returns immediately. The remote run.sh — and ONLY it — writes
    status.json/done/stderr.log/transcript back into that same dir; this side
    writes neither, so there is no Syncthing write-write conflict. status() reads
    the synced run-state locally (no SSH to poll).

    `lane` selects the relay lane ("prompts"=build, "recon"=recon): staged prompt
    read from <to_seat>/<lane>/latest.md, return paths under <from_seat>/<lane>/.
    `model` of None means the CLI default (no --model flag).

    `remote_vault`, when set, is the target host's OWN absolute path to the same
    Syncthing-synced vault (e.g. the Mac's "/Users/davidgomez/Vaults/homelab-vault" vs. worker2's
    "/home/david/Vaults/homelab-vault" — same tree,
    different home root because the account name differs). worker1/worker3/localworker all run
    as user "david" under "/home/david", coincidentally identical to worker2's VAULT,
    so they never needed this. Anything baked into run.sh or the ssh wait-loop
    command (which EXECUTE on the target host) must use the remote_vault-rooted
    path; anything worker2 itself writes to or reads back from (prompt_path,
    run_sh_path, the returned transcript path, status() polling) stays on the
    local VAULT-rooted path, since Syncthing mirrors file CONTENT across both
    roots, not the path string itself.
    Returns {ok:True, token, lane, started_at, transcript, note} on launch, else
    {ok:False, error, ...}. Refuses if `token` not in the staged prompt.

    The success marker is CONDITIONAL (2026-07-13, FLEET-WORKER2-BUILD-20260713-
    launch-marker): for the ssh branch, after backgrounding the launch we poll
    briefly (~3s) for `<RUNDIR>/status.json` (run.sh's first write) before echoing
    `marker`; if it never appears we echo the paired `<SEAT>_LAUNCH_FAILED` marker
    instead and this returns {ok:False, error:"launch failed", stderr,
    launch_stderr_path} — the launch command's own stderr is captured to
    `<RUNDIR>/launch.stderr` (no longer swallowed) so a dead launch (e.g. a missing
    binary on the target) is diagnosable instead of being silently reported as
    success. (An earlier draft of this fix also treated the backgrounded PID
    staying alive as an alternate success signal; dropped after an empirical
    negative-test run showed the setsid/nohup wrapper PID can outlive an
    instantly-failed exec by ~1s, which would have produced false positives.)
    The local (worker2, systemd-run) branch performs an analogous local poll for
    status.json before reporting success.
    """
    token = (token or "").strip()
    if not token:
        return {"ok": False, "error": "empty token"}
    if not _LAUNCH_TOKEN_RE.fullmatch(token):
        return {"ok": False, "error": "malformed token"}

    if lane not in LANES:
        return {"ok": False, "error": "bad lane"}
    try:
        max_runtime = _normalize_max_runtime(max_runtime_seconds)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    staged_prompt = os.path.join(VAULT, to_seat, lane, "latest.md")
    response_path = os.path.join(VAULT, from_seat, lane, "latest_response.md")
    responses_log = os.path.join(VAULT, from_seat, lane, "responses.md")

    if prompt_content is None:
        try:
            with open(staged_prompt, "r", encoding="utf-8", errors="replace") as f:
                staged = f.read()
        except Exception as e:
            return {
                "ok": False,
                "error": f"cannot read staged prompt: {type(e).__name__}: {e}",
            }
    elif not isinstance(prompt_content, str) or not prompt_content.strip():
        return {"ok": False, "error": "invalid prompt content"}
    else:
        staged = prompt_content

    if token not in staged:
        return {
            "ok": False,
            "error": "token mismatch",
            "staged_hint": _first_token_like_line(staged),
        }

    started_at = _now()
    run_dir = os.path.join(runs_root, token)
    # A run token is a permanent, single-use idempotency key.  The claim must
    # be made atomically before any per-run file is written so concurrent MCP
    # worker threads cannot both launch the same token and race on run state.
    # Never remove this directory after a later launch failure: the durable
    # failure record is part of the token's history and retries need a new token.
    try:
        os.makedirs(run_dir, exist_ok=False)
    except FileExistsError:
        return {
            "ok": False,
            "error": "token already launched",
            "token": token,
            "seat": seat,
            "lane": lane,
        }
    os.makedirs(transcripts_dir, exist_ok=True)

    if worker_routing is not None:
        route_path = os.path.join(run_dir, "worker-routing.json")
        try:
            with open(route_path, "x", encoding="utf-8") as handle:
                json.dump(worker_routing, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(route_path, 0o600)
        except Exception as exc:
            _mark_launch_failed(
                run_dir, f"worker routing persistence failed: {type(exc).__name__}"
            )
            return {
                "ok": False,
                "error": f"cannot persist worker routing: {type(exc).__name__}",
            }

    # Preserve the caller's exact source prompt in the claimed token directory.
    # This path is created once, never reopened for writing, and made read-only;
    # prompt.txt remains the executable prompt with the delivery footer appended.
    request_path = os.path.join(run_dir, "request.md")
    try:
        with open(request_path, "x", encoding="utf-8") as f:
            f.write(staged)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(request_path, 0o444)
    except Exception as e:
        _mark_launch_failed(run_dir, f"request persistence failed: {type(e).__name__}")
        return {"ok": False, "error": f"cannot persist immutable request: {type(e).__name__}"}

    deadline_runner_path = os.path.join(run_dir, "deadline_runner.py")
    if max_runtime:
        try:
            with open(deadline_runner_path, "x", encoding="utf-8") as handle:
                handle.write(_DEADLINE_RUNNER)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(deadline_runner_path, 0o755)
        except Exception as exc:
            _mark_launch_failed(run_dir, f"deadline runner persistence failed: {type(exc).__name__}")
            return {"ok": False, "error": f"cannot persist deadline runner: {type(exc).__name__}"}

    # All cloud providers emit JSONL. Gemini previously used plain text, which
    # made its runs invisible to Hero's Path.
    transcript_ext = "json"
    transcript = os.path.join(transcripts_dir, f"{token}.{transcript_ext}")
    prompt_path = os.path.join(run_dir, "prompt.txt")
    run_sh_path = os.path.join(run_dir, "run.sh")

    token_response_path = os.path.join(run_dir, "response.md")

    # Remote-side path variants — what run.sh and the footer instructions must
    # use since they execute/apply on the TARGET host, not alpha. Identity when
    # remote_vault is unset (the worker1/worker3/localworker/worker2 seats all share the Tower
    # host's VAULT root).
    def _on_target(p: str) -> str:
        return p.replace(VAULT, remote_vault, 1) if remote_vault and p.startswith(VAULT) else p

    target_run_dir = _on_target(run_dir)
    target_transcripts_dir = _on_target(transcripts_dir)
    target_response_path = _on_target(response_path)
    target_responses_log = _on_target(responses_log)
    target_token_response_path = _on_target(token_response_path)

    target_session_note = _on_target(_resolve_session_note(session_note))
    launcher_owns_delivery = (
        provider in ("claude", "codex", "gemini")
        and footer_fn in (
            _build_footer,
            _build_worker3_footer,
            _build_worker2_footer,
            _build_worker4_footer,
        )
    )
    executable_prompt = _append_worker_execution_context(
        _append_worker_artifact_contract(staged), execution_context
    )
    if launcher_owns_delivery:
        # The structured CLI stream is the delivery boundary. Do not ask a model
        # to interpolate its Markdown into shell commands or spend extra tool
        # turns maintaining relay projections. The launcher extracts the final
        # response and publishes it with Python file I/O for every cloud provider.
        full_prompt = executable_prompt + "\n".join([
            "",
            "",
            "---",
            "",
            "## RELAY DELIVERY (automated by the cloud-worker launcher)",
            "",
            "Do NOT manually write response.md, latest_response.md, responses.md, "
            "or the relay session note. Return the complete final report only through "
            "your normal final response; the launcher publishes it atomically and "
            "appends the worker execution record.",
            f"End the final response with the token `{token}`.",
        ])
    else:
        selected_footer_fn = footer_fn
        if (
            footer_fn is _build_localworker_footer
            and "FLEET_COMPACT_DELIVERY_V1_BEGIN" in staged
            and "FLEET_COMPACT_DELIVERY_V1_END" in staged
        ):
            selected_footer_fn = _build_localworker_compact_footer
        full_prompt = executable_prompt + selected_footer_fn(
            token, session_note, target_response_path, target_responses_log, target_token_response_path
        )
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(full_prompt)

    # run.sh runs ON THE TARGET HOST. Built by str.replace on __PLACEHOLDERS__
    # (NOT an f-string — the printf payload is JSON with literal braces). Only
    # this script, on the target, writes status.json/done; alpha writes neither.
    run_sh_template = (
        "#!/usr/bin/env bash\n"
        "set +e\n"
        # Tag this relay run so the sessions-viewer hook layer on the target host
        # reports it as relay-<seat> (interactive keeps the dispatcher default).
        "export CLAUDE_SEAT=__SEATENV__\n"
        'RUNDIR="__RUNDIR__"; TR="__TR__"; mkdir -p "$RUNDIR" "$TR"\n'
        'ST="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
        "printf '{\"token\":\"__TOK__\",\"lane\":\"__LANE__\",\"pid\":%s,"
        "\"started_at\":\"%s\",\"status\":\"running\",\"transcript\":"
        "\"__TR__/__TOK__.json\",\"response_path\":\"__RESP__\",\"source\":\"__SEAT__\","
        "\"provider\":\"__PROVIDER__\",\"provider_reason\":\"__PROVIDER_REASON__\","
        "\"model\":\"__MODEL__\",\"max_runtime_seconds\":__MAX_RUNTIME__}"
        "\\n' \"$$\" \"$ST\" > \"$RUNDIR/status.json\"\n"
        + _HEARTBEAT_SNIPPET +
        'if [ "__MAX_RUNTIME__" -gt 0 ]; then\n'
        '  "__PYTHON__" "$RUNDIR/deadline_runner.py" --max-runtime "__MAX_RUNTIME__" '
        '--terminal "$RUNDIR/terminal.json" -- "__CLAUDE__" __MODELFLAG__-p '
        '--output-format stream-json --verbose --dangerously-skip-permissions '
        '< "$RUNDIR/prompt.txt" > "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
        '  RC=$?\n'
        'else\n'
        '  "__CLAUDE__" __MODELFLAG__-p --output-format stream-json --verbose '
        '--dangerously-skip-permissions < "$RUNDIR/prompt.txt" '
        '> "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
        '  RC=$?\n'
        'worker1\n'
        'echo "$RC" > "$RUNDIR/done"\n'
    )
    if provider == "codex":
        run_sh_template = (
            "#!/usr/bin/env bash\n"
            "set +e\n"
            "export FLEET_SEAT=__SEATENV__\n"
            "export CODEX_HOME=__CODEX_HOME__\n"
            'RUNDIR="__RUNDIR__"; TR="__TR__"; mkdir -p "$RUNDIR" "$TR"\n'
            'ST="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
            "printf '{\"token\":\"__TOK__\",\"lane\":\"__LANE__\",\"pid\":%s,"
            "\"started_at\":\"%s\",\"status\":\"running\",\"transcript\":"
            "\"__TR__/__TOK__.json\",\"response_path\":\"__RESP__\",\"source\":\"__SEAT__\","
            "\"provider\":\"__PROVIDER__\",\"provider_reason\":\"__PROVIDER_REASON__\","
            "\"model\":\"__MODEL__\",\"max_runtime_seconds\":__MAX_RUNTIME__}"
            "\\n' \"$$\" \"$ST\" > \"$RUNDIR/status.json\"\n"
            + _HEARTBEAT_SNIPPET +
            'cd "$HOME"\n'
            'if [ "__MAX_RUNTIME__" -gt 0 ]; then\n'
            '  "__PYTHON__" "$RUNDIR/deadline_runner.py" --max-runtime "__MAX_RUNTIME__" '
            '--terminal "$RUNDIR/terminal.json" -- "__CLAUDE__" exec --json '
            '--skip-git-repo-check -m "__MODEL__" -s danger-full-access - '
            '< "$RUNDIR/prompt.txt" > "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
            '  RC=$?\n'
            'else\n'
            '  "__CLAUDE__" exec --json --skip-git-repo-check -m "__MODEL__" '
            '-s danger-full-access - < "$RUNDIR/prompt.txt" '
            '> "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
            '  RC=$?\n'
            'worker1\n'
            'echo "$RC" > "$RUNDIR/done"\n'
        )
    if provider == "gemini":
        run_sh_template = (
            "#!/usr/bin/env bash\n"
            "set +e\n"
            "export FLEET_SEAT=__SEATENV__\n"
            'RUNDIR="__RUNDIR__"; TR="__TR__"; mkdir -p "$RUNDIR" "$TR"\n'
            'ST="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
            "printf '{\"token\":\"__TOK__\",\"lane\":\"__LANE__\",\"pid\":%s,"
            "\"started_at\":\"%s\",\"status\":\"running\",\"transcript\":"
            "\"__TR__/__TOK__.json\",\"response_path\":\"__RESP__\",\"source\":\"__SEAT__\","
            "\"provider\":\"__PROVIDER__\",\"provider_reason\":\"__PROVIDER_REASON__\","
            "\"model\":\"__MODEL__\",\"max_runtime_seconds\":__MAX_RUNTIME__}"
            "\\n' \"$$\" \"$ST\" > \"$RUNDIR/status.json\"\n"
            + _HEARTBEAT_SNIPPET +
            'cd "$HOME"\n'
            'PROMPT="$(cat "$RUNDIR/prompt.txt")"\n'
            'if [ "__MAX_RUNTIME__" -gt 0 ]; then\n'
            '  "__PYTHON__" "$RUNDIR/deadline_runner.py" --max-runtime "__MAX_RUNTIME__" '
            '--terminal "$RUNDIR/terminal.json" -- "__CLAUDE__" --agent fleet-worker '
            '--model "__MODEL__" --effort high --mode accept-edits '
            '--dangerously-skip-permissions --output-format stream-json '
            '--print-timeout "__MAX_RUNTIME__s" --print "$PROMPT" '
            '> "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
            '  RC=$?\n'
            'else\n'
            '  "__CLAUDE__" --agent fleet-worker --model "__MODEL__" --effort high '
            '--mode accept-edits --dangerously-skip-permissions '
            '--output-format stream-json --print-timeout "168h" --print "$PROMPT" '
            '> "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
            '  RC=$?\n'
            'worker1\n'
            'echo "$RC" > "$RUNDIR/done"\n'
        )
    if seat == "worker4":
        # caffeinate -i holds off idle/display sleep for the run's duration.
        # It does NOT prevent lid-close suspend on battery power — an
        # operational caveat (keep the lid open), not fixable from here.
        run_sh_template = (
            "#!/usr/bin/env bash\n"
            "set +e\n"
            "export CLAUDE_SEAT=__SEATENV__\n"
            'RUNDIR="__RUNDIR__"; TR="__TR__"; mkdir -p "$RUNDIR" "$TR"\n'
            'ST="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
            "printf '{\"token\":\"__TOK__\",\"lane\":\"__LANE__\",\"pid\":%s,"
            "\"started_at\":\"%s\",\"status\":\"running\",\"transcript\":"
            "\"__TR__/__TOK__.json\",\"response_path\":\"__RESP__\",\"source\":\"__SEAT__\","
            "\"model\":\"__MODEL__\",\"max_runtime_seconds\":__MAX_RUNTIME__}"
            "\\n' \"$$\" \"$ST\" > \"$RUNDIR/status.json\"\n"
            + _HEARTBEAT_SNIPPET +
            'export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$HOME/.config/fleet/worker4-oauth-token" 2>/dev/null)"\n'
            "unset ANTHROPIC_API_KEY\n"
            'if [ "__MAX_RUNTIME__" -gt 0 ]; then\n'
            '  "__PYTHON__" "$RUNDIR/deadline_runner.py" --max-runtime "__MAX_RUNTIME__" '
            '--terminal "$RUNDIR/terminal.json" -- caffeinate -i "__CLAUDE__" '
            '__MODELFLAG__-p --output-format stream-json --verbose '
            '--dangerously-skip-permissions < "$RUNDIR/prompt.txt" '
            '> "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
            '  RC=$?\n'
            'else\n'
            '  caffeinate -i "__CLAUDE__" __MODELFLAG__-p --output-format stream-json --verbose '
            '--dangerously-skip-permissions < "$RUNDIR/prompt.txt" '
            '> "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
            '  RC=$?\n'
            'worker1\n'
            'echo "$RC" > "$RUNDIR/done"\n'
        )
    elif seat == "localworker":
        run_sh_template = (
            "#!/usr/bin/env bash\n"
            "set +e\n"
            "export CLAUDE_SEAT=__SEATENV__\n"
            'RUNDIR="__RUNDIR__"; TR="__TR__"; mkdir -p "$RUNDIR" "$TR"\n'
            'ST="$(date -u +%Y-%m-%dT%H:%M:%SZ)"\n'
            "printf '{\"token\":\"__TOK__\",\"lane\":\"__LANE__\",\"pid\":%s,"
            "\"started_at\":\"%s\",\"status\":\"running\",\"transcript\":"
            "\"__TR__/__TOK__.json\",\"response_path\":\"__RESP__\",\"source\":\"__SEAT__\","
            "\"model\":\"__MODEL__\",\"max_runtime_seconds\":__MAX_RUNTIME__}"
            "\\n' \"$$\" \"$ST\" > \"$RUNDIR/status.json\"\n"
            + _HEARTBEAT_SNIPPET +
            'if [ "__MAX_RUNTIME__" -gt 0 ]; then\n'
            '  "__PYTHON__" "$RUNDIR/deadline_runner.py" --max-runtime "__MAX_RUNTIME__" '
            '--terminal "$RUNDIR/terminal.json" -- '
            '/home/david/repos/fleet-agent/.venv/bin/python '
            '/home/david/repos/fleet-agent/fleet_agent.py '
            '__MODELFLAG__--prompt "$RUNDIR/prompt.txt" '
            '--run-dir "$RUNDIR" --token "__TOK__" --lane "__LANE__" '
            '--response-path "__RESP__" --responses-log '
            '"/home/david/Vaults/homelab-vault/from-localworker/__LANE__/responses.md" '
            '--session-note "" > "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
            '  RC=$?\n'
            'else\n'
            '  /home/david/repos/fleet-agent/.venv/bin/python '
            '/home/david/repos/fleet-agent/fleet_agent.py '
            '__MODELFLAG__--prompt "$RUNDIR/prompt.txt" --run-dir "$RUNDIR" --token '
            '"__TOK__" --lane "__LANE__" --response-path "__RESP__" '
            '--responses-log "/home/david/Vaults/homelab-vault/from-localworker/__LANE__/responses.md" '
            '--session-note "" > "$TR/__TOK__.json" 2> "$RUNDIR/stderr.log"\n'
            '  RC=$?\n'
            'worker1\n'
            'echo "$RC" > "$RUNDIR/done"\n'
        )
    if launcher_owns_delivery:
        delivery_template = (
            '# Publish model-authored content using Python file I/O only. The final '
            '# response never becomes shell source.\n'
            'if [ "$RC" -eq 0 ] && [ -s "$TR/__TOK__.json" ]; then\n'
            '  "__PYTHON__" - "$TR/__TOK__.json" "$RUNDIR/response.md" '
            '"__RESP__" "__RESPLOG__" "__SESSION_NOTE__" "__TOK__" "__SEAT__" '
            '"__PROVIDER__" <<\'PY\'\n'
            'import json, os, sys, tempfile\n'
            'source, token_response, latest_response, responses_log, session_note, token, seat, provider = sys.argv[1:]\n'
            'answer = ""\n'
            'provider_error = False\n'
            'with open(source, "r", encoding="utf-8", errors="replace") as handle:\n'
            '    for line in handle:\n'
            '        try:\n'
            '            record = json.loads(line)\n'
            '        except Exception:\n'
            '            continue\n'
            '        if provider == "gemini" and record.get("event") == "result":\n'
            '            result = record.get("result") or {}\n'
            '            candidate = result.get("response")\n'
            '            provider_error = result.get("status") not in (None, "SUCCESS")\n'
            '        elif provider == "codex" and record.get("type") == "item.completed":\n'
            '            item = record.get("item") or {}\n'
            '            candidate = item.get("text") if item.get("type") == "agent_message" else None\n'
            '        elif provider == "claude" and record.get("type") == "result":\n'
            '            candidate = record.get("result")\n'
            '            provider_error = bool(record.get("is_error"))\n'
            '        else:\n'
            '            candidate = None\n'
            '        if isinstance(candidate, str) and candidate.strip():\n'
            '            answer = candidate\n'
            'if provider_error:\n'
            '    raise SystemExit(f"{provider} result reported an error")\n'
            'if not answer.strip():\n'
            '    raise SystemExit(f"missing nonempty {provider} final response")\n'
            'def atomic_write(path, content):\n'
            '    parent = os.path.dirname(path)\n'
            '    os.makedirs(parent, exist_ok=True)\n'
            '    fd, tmp = tempfile.mkstemp(prefix=".response.", dir=parent, text=True)\n'
            '    try:\n'
            '        with os.fdopen(fd, "w", encoding="utf-8") as out:\n'
            '            out.write(content)\n'
            '            out.flush()\n'
            '            os.fsync(out.fileno())\n'
            '        os.replace(tmp, path)\n'
            '    finally:\n'
            '        if os.path.exists(tmp):\n'
            '            os.unlink(tmp)\n'
            'atomic_write(token_response, answer)\n'
            'atomic_write(latest_response, answer)\n'
            'os.makedirs(os.path.dirname(responses_log), exist_ok=True)\n'
            'with open(responses_log, "a", encoding="utf-8") as out:\n'
            '    out.write("\\n---\\n" + answer)\n'
            '    out.flush()\n'
            '    os.fsync(out.fileno())\n'
            'if session_note:\n'
            '    os.makedirs(os.path.dirname(session_note), exist_ok=True)\n'
            '    with open(session_note, "a", encoding="utf-8") as out:\n'
            '        out.write(f"\\n\\n### [{seat.title()}] {token}\\n\\n{answer}\\n")\n'
            '        out.flush()\n'
            '        os.fsync(out.fileno())\n'
            'PY\n'
            '  PUB_RC=$?\n'
            '  if [ "$PUB_RC" -ne 0 ]; then RC=70; worker1\n'
            'worker1\n'
        )
        run_sh_template = run_sh_template.replace(
            'echo "$RC" > "$RUNDIR/done"\n',
            delivery_template + 'echo "$RC" > "$RUNDIR/done"\n',
            1,
        )

    run_sh = (
        run_sh_template
        .replace("__RUNDIR__", target_run_dir)
        .replace("__TR__", target_transcripts_dir)
        .replace("__PYTHON__", "/opt/homebrew/bin/python3" if seat == "worker4" else "/usr/bin/python3")
        .replace("__MAX_RUNTIME__", str(max_runtime))
        .replace("__RESP__", target_response_path)
        .replace("__RESPLOG__", target_responses_log)
        .replace("__SESSION_NOTE__", target_session_note)
        .replace("__CLAUDE__", claude_bin)
        .replace("__CODEX_HOME__", CODEX_HOME_WORKER)
        .replace("__MODELFLAG__", f"--model {model} " if model else "")
        .replace("__MODEL__", model if model else "none (custom agent)")
        .replace("__PROVIDER_REASON__", provider_reason)
        .replace("__PROVIDER__", provider)
        .replace("__SEATENV__", seat_env)
        .replace("__SEAT__", seat)
        .replace("__TOK__", token)
        .replace("__LANE__", lane)
    )
    with open(run_sh_path, "w", encoding="utf-8") as f:
        f.write(run_sh)

    # Remote launch. The wait loop covers the alpha->target Syncthing race for the
    # two files we just wrote (~1s typical, 20s cap), THEN setsid+nohup detaches the
    # run so the ssh returns immediately (recon-validated).
    # macOS has no `setsid` (it's util-linux, Linux-only) — on worker4 it silently
    # no-op'd the whole `setsid nohup ... &` pipeline (command-not-found) UNTIL
    # 2026-07-13 (FLEET-WORKER2-BUILD-20260713-launch-marker): the launch stderr
    # used to be swallowed (`2>&1 >/dev/null`, redirecting stderr to a stdout that
    # was already pointed at /dev/null) and the marker echoed unconditionally
    # regardless of whether run.sh actually started, so a dead launch still
    # reported success. Now: launch stderr goes to `$RUNDIR/launch.stderr`
    # (diagnosable), and the marker is CONDITIONAL — after backgrounding, we poll
    # up to ~3s for `$RUNDIR/status.json` (run.sh's first write) to appear; only
    # then do we echo the success marker, else a distinct `<fail_marker>` so the
    # caller can tell real launch failures from a working one. Confirmed empirically
    # 2026-07-13: `nohup ... &` alone detaches fine on macOS (ssh returns
    # immediately, the child survives the session ending) — no setsid needed there.
    launch_prefix = "nohup" if seat == "worker4" else "setsid nohup"
    # marker is always "<SEAT>_LAUNCHED" today; derive the paired failure marker
    # from it so callers can distinguish "run.sh never started" from success.
    fail_marker = (
        marker.replace("_LAUNCHED", "_LAUNCH_FAILED") if marker.endswith("_LAUNCHED")
        else f"{marker}_FAILED"
    )
    launch_stderr_path = os.path.join(run_dir, "launch.stderr")
    target_launch_stderr_path = _on_target(launch_stderr_path)
    # NOTE: deliberately status.json-ONLY, no "OR kill -0 $LPID" fallback. Empirically
    # verified (negative selftest, 2026-07-13) that `kill -0` on the backgrounded
    # setsid/nohup wrapper's PID can report "alive" for up to ~1s AFTER the wrapped
    # `bash run.sh` has already failed to exec (e.g. missing file) — setsid can
    # fork and the original PID lingers briefly tearing down even though the real
    # child never started. That false-aliveness would silently defeat the whole
    # point of this hardening. status.json is unambiguous: run.sh writes it as its
    # very first action, so its presence means run.sh definitely started.
    required_files = (
        "[ -f '__RUNDIR__/run.sh' ] && [ -f '__RUNDIR__/prompt.txt' ]"
        + (" && [ -f '__RUNDIR__/deadline_runner.py' ]" if max_runtime else "")
    ).replace("__RUNDIR__", target_run_dir)
    remote_cmd = (
        "for i in $(seq 1 40); do "
        "__REQUIRED__ && break; "
        "sleep 0.5; done; "
        "__LAUNCH__ bash '__RUNDIR__/run.sh' </dev/null >/dev/null 2>'__LAUNCHERR__' & "
        "ok=0; "
        "for i in $(seq 1 6); do "
        "[ -f '__RUNDIR__/status.json' ] && ok=1 && break; "
        "sleep 0.5; done; "
        "if [ \"$ok\" = 1 ]; then echo __MARKER__; else echo __FAILMARKER__; worker1"
    ).replace("__REQUIRED__", required_files).replace("__LAUNCH__", launch_prefix).replace("__RUNDIR__", target_run_dir) \
     .replace("__LAUNCHERR__", target_launch_stderr_path) \
     .replace("__MARKER__", marker).replace("__FAILMARKER__", fail_marker)

    if local:
        # LOCAL seat (alpha): no ssh, no Syncthing race (files are on local disk),
        # so skip the wait-loop. systemd-run --user launches run.sh as its OWN
        # transient user unit -> detached AND outside the MCP's cgroup, so an MCP
        # restart cannot kill an in-flight run. --collect reaps it after exit.
        try:
            r = subprocess.run(
                ["systemd-run", "--user", "--quiet", "--collect", "--setenv=PATH=/home/david/.local/bin:/usr/local/bin:/usr/bin:/bin", "--setenv=HOME=/home/david",
                 "bash", run_sh_path],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode != 0:
                _mark_launch_failed(run_dir, f"systemd-run rc={r.returncode}")
                return {"ok": False, "error": "local launch failed",
                        "stderr": (r.stderr or "").strip()[-500:]}
        except Exception as e:
            _mark_launch_failed(run_dir, f"local launch exception: {type(e).__name__}")
            return {"ok": False, "error": f"local launch exception: {type(e).__name__}: {e}"}

        # systemd-run succeeding only means the transient unit was accepted —
        # confirm run.sh actually started (its first action is writing
        # status.json) before reporting success. No Syncthing race here (local
        # disk), so a short poll suffices.
        status_json = os.path.join(run_dir, "status.json")
        verified = False
        for _ in range(6):
            if os.path.exists(status_json):
                verified = True
                break
            time.sleep(0.5)
        if not verified:
            _mark_launch_failed(run_dir, "local status.json never appeared")
            return {
                "ok": False,
                "error": "local launch verification failed: status.json never appeared",
                "run_dir": run_dir,
            }
    else:
        def _ssh_launch(target: str):
            return subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote_cmd],
                capture_output=True,
                text=True,
                timeout=40,
            )

        def _read_launch_stderr() -> str:
            # Best-effort: launch.stderr is written on the TARGET host and only
            # visible here once Syncthing carries it back — may not have synced
            # yet at return time, in which case we fall back to ssh's own stderr.
            try:
                with open(launch_stderr_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read().strip()[-1000:]
            except Exception:
                return ""

        try:
            r = _ssh_launch(host)
            if fail_marker in r.stdout:
                _mark_launch_failed(run_dir, "ssh LAUNCH_FAILED marker")
                return {
                    "ok": False,
                    "error": "launch failed",
                    "stderr": _read_launch_stderr() or (r.stderr or "").strip()[-500:],
                    "launch_stderr_path": launch_stderr_path,
                }
            if marker not in r.stdout or r.returncode != 0:
                _mark_launch_failed(run_dir, "ssh rc!=0 / marker missing")
                return {
                    "ok": False,
                    "error": "ssh launch failed",
                    "stderr": (r.stderr or "").strip()[-500:],
                }
        except Exception as e:
            _mark_launch_failed(run_dir, f"ssh exception: {type(e).__name__}")
            return {"ok": False, "error": f"ssh launch exception: {type(e).__name__}: {e}"}

    return {
        "ok": True,
        "token": token,
        "lane": lane,
        "started_at": started_at,
        "transcript": transcript,
        "request_prompt": request_path,
        "provider": provider,
        "provider_reason": provider_reason,
        "model": model,
        "note": launched_note,
    }


def kick_delta(token: str, lane: str = "prompts", session_note: str = "", prompt_content: str = None, max_runtime_seconds: int = 0, provider: str = "auto", task_size: str = "small", execution_context: str = "", worker_routing: dict = None) -> dict:
    """Launch a detached headless `claude -p` run on Worker1 (delta, REMOTE).

    worker2 writes the prompt + a run.sh into WORKER1_RUNS/<token>/ (which Syncthing
    carries to delta), then SSHes delta and `setsid nohup bash run.sh`.
    The remote run.sh self-writes run-state back into that same dir; status()
    reads the synced run-state worker2-side (no SSH to poll). Worker1 is pinned to
    Sonnet 5 via WORKER1_MODEL.
    """
    spec = _entry_provider_spec(
        "delta", provider, DELTA_RUNS, token, lane, max_runtime_seconds, prompt_content,
        task_size,
    )
    if not spec["ok"]:
        return spec
    result = _kick_remote(
        token,
        lane,
        session_note,
        seat="delta",
        to_seat="to-delta",
        from_seat="from-worker1",
        runs_root=DELTA_RUNS,
        transcripts_dir=DELTA_TRANSCRIPTS,
        claude_bin=spec["bin"] or CLAUDE_BIN_WORKER1,
        model=spec["model"] or WORKER1_MODEL,
        host=WORKER1_HOST,
        host_ip=None,  # WORKER1_HOST_IP retired under DL-15 / SSH-1; host_ip is no longer read by _kick_remote
        seat_env="relay-worker1",
        footer_fn=_build_footer,
        marker="WORKER1_LAUNCHED",
        prompt_content=prompt_content,
        max_runtime_seconds=max_runtime_seconds,
        provider=spec["provider"],
        provider_reason=spec["reason"],
        execution_context=execution_context,
        worker_routing=worker_routing,
        launched_note="launched on delta (detached)",
    )
    return _finish_routed_launch(result, spec, WORKER1_RUNS, token)


def kick_charlie(token: str, lane: str = "prompts", session_note: str = "", prompt_content: str = None, max_runtime_seconds: int = 0, provider: str = "auto", task_size: str = "small", execution_context: str = "", worker_routing: dict = None) -> dict:
    """Launch a detached headless `claude -p` run on Worker3 (charlie, REMOTE).

    Same mechanism as kick_delta (see _kick_remote), pointed at the to-worker3/from-worker3
    lanes and pinned to WORKER3_MODEL.
    """
    spec = _entry_provider_spec(
        "charlie", provider, CHARLIE_RUNS, token, lane, max_runtime_seconds, prompt_content,
        task_size,
    )
    if not spec["ok"]:
        return spec
    result = _kick_remote(
        token,
        lane,
        session_note,
        seat="charlie",
        to_seat="to-charlie",
        from_seat="from-worker3",
        runs_root=CHARLIE_RUNS,
        transcripts_dir=CHARLIE_TRANSCRIPTS,
        claude_bin=spec["bin"] or CLAUDE_BIN_WORKER3,
        model=spec["model"] or WORKER3_MODEL,
        host=WORKER3_HOST,
        host_ip=None,  # WORKER3_HOST_IP retired under DL-15 / SSH-1; host_ip is no longer read by _kick_remote
        seat_env="relay-worker3",
        footer_fn=_build_worker3_footer,
        marker="WORKER3_LAUNCHED",
        prompt_content=prompt_content,
        max_runtime_seconds=max_runtime_seconds,
        provider=spec["provider"],
        provider_reason=spec["reason"],
        execution_context=execution_context,
        worker_routing=worker_routing,
        launched_note="launched on charlie (detached)",
    )
    return _finish_routed_launch(result, spec, WORKER3_RUNS, token)


def kick_localworker(token: str, lane: str = "prompts", session_note: str = "", prompt_content: str = None, max_runtime_seconds: int = 0, execution_context: str = "", worker_routing: dict = None) -> dict:
    """Launch a detached headless run on Localworker (charlie, REMOTE)."""
    return _kick_remote(
        token,
        lane,
        session_note,
        seat="localworker",
        to_seat="to-localworker",
        from_seat="from-localworker",
        runs_root=LOCALWORKER_RUNS,
        transcripts_dir=LOCALWORKER_TRANSCRIPTS,
        claude_bin=CLAUDE_BIN_LOCALWORKER,
        model=LOCALWORKER_MODEL,
        host=LOCALWORKER_HOST,
        host_ip=None,  # LOCALWORKER_HOST_IP retired under DL-15 / SSH-1; host_ip is no longer read by _kick_remote
        seat_env="relay-localworker",
        footer_fn=_build_localworker_footer,
        marker="LOCALWORKER_LAUNCHED",
        prompt_content=prompt_content,
        max_runtime_seconds=max_runtime_seconds,
        execution_context=execution_context,
        worker_routing=worker_routing,
        launched_note="launched on charlie (detached)",
    )


def kick_worker4(token: str, lane: str = "prompts", session_note: str = "", prompt_content: str = None, max_runtime_seconds: int = 0) -> dict:
    """Launch a detached headless `claude -p` run on Worker4 (MacBook, REMOTE).

    Same mechanism as kick_delta/kick_charlie (see _kick_remote), pointed at the
    to-worker4/from-worker4 lanes and pinned to WORKER4_MODEL. The launched command is
    wrapped in `caffeinate -i` (see _kick_remote's seat == "worker4" branch) to
    hold off idle/display sleep for the run's duration; it does NOT prevent
    lid-close suspend on battery power.
    """
    return _kick_remote(
        token,
        lane,
        session_note,
        seat="worker4",
        to_seat="to-worker4",
        from_seat="from-worker4",
        runs_root=WORKER4_RUNS,
        transcripts_dir=WORKER4_TRANSCRIPTS,
        claude_bin=CLAUDE_BIN_WORKER4,
        model=WORKER4_MODEL,
        host=WORKER4_HOST,
        host_ip=None,  # WORKER4_HOST_IP retired under DL-15 / SSH-1; host_ip is no longer read by _kick_remote
        seat_env="relay-worker4",
        footer_fn=_build_worker4_footer,
        marker="WORKER4_LAUNCHED",
        launched_note="launched on macbook (detached)",
        remote_vault=WORKER4_VAULT,
        prompt_content=prompt_content,
        max_runtime_seconds=max_runtime_seconds,
    )


def kick_alpha(token: str, lane: str = "prompts", session_note: str = "", prompt_content: str = None, max_runtime_seconds: int = 0, provider: str = "auto", task_size: str = "small", execution_context: str = "", worker_routing: dict = None) -> dict:
    """Detached headless worker on Worker2's physical host (Alpha).

    The primary Tower runs on Alpha and uses the local systemd-run branch. A
    promoted Standby Tower runs on Bravo and must use the normal SSH
    branch instead. TOWER_NODE is set explicitly by each service unit;
    this avoids self-SSH on primary and prevents standby from launching a worker
    on the wrong computer.
    """
    # WORKER1-2: TOWER_NODE may be either the pre- or post-rename node name.
    worker2_is_local = canonical_node(TOWER_NODE) == "alpha"
    spec = _entry_provider_spec(
        "alpha", provider, ALPHA_RUNS, token, lane, max_runtime_seconds, prompt_content,
        task_size,
    )
    if not spec["ok"]:
        return spec
    result = _kick_remote(
        token,
        lane,
        session_note,
        seat="alpha",
        to_seat="to-alpha",
        from_seat="from-worker2",
        runs_root=ALPHA_RUNS,
        transcripts_dir=ALPHA_TRANSCRIPTS,
        claude_bin=spec["bin"] or CLAUDE_BIN_WORKER2,
        model=spec["model"] or WORKER2_MODEL,
        host=None if worker2_is_local else WORKER2_HOST,
        host_ip=None,
        seat_env="relay-worker2",
        footer_fn=_build_worker2_footer,
        marker="WORKER2_LAUNCHED",
        launched_note=(
            "launched on worker2 (local, detached)"
            if worker2_is_local
            else "launched on worker2 via alpha SSH (detached)"
        ),
        local=worker2_is_local,
        prompt_content=prompt_content,
        max_runtime_seconds=max_runtime_seconds,
        provider=spec["provider"],
        provider_reason=spec["reason"],
        execution_context=execution_context,
        worker_routing=worker_routing,
    )
    return _finish_routed_launch(result, spec, WORKER2_RUNS, token)


def _remote_status(token: str, runs_root: str, transcripts_dir: str, from_seat: str, source: str) -> dict:
    """Poll a remote run by token. Reads only the (vault-synced) filesystem.

    NO /proc liveness — the pid in status.json belongs to the target host,
    meaningless against worker2's /proc — so we trust the run.sh-written status.json,
    the heartbeat file it refreshes every ~30s, and the `done` sentinel. States:
    run dir absent -> missing; dir present but status.json not yet synced ->
    "launching" (or "stale"/launch_timeout past TOWER_LAUNCH_GRACE_SECONDS);
    `done` absent -> stored status ("running", or "stale"/heartbeat_timeout past
    TOWER_STALE_AFTER_SECONDS since the heartbeat/status.json mtime); `done`
    present -> "done" with exit_code/transcript_bytes/response_token_match
    (authoritative — never reclassified as stale).
    """
    token = (token or "").strip()
    run_dir = os.path.join(runs_root, token)
    if not os.path.isdir(run_dir):
        return {"ok": False, "status": "missing"}

    # A terminal done sentinel is authoritative even when run.sh never started
    # and therefore never wrote status.json (the launch-failure marker path).
    done_path = os.path.join(run_dir, "done")
    status_json = os.path.join(run_dir, "status.json")
    if not os.path.exists(done_path) and not os.path.exists(status_json):
        # run.sh on the target hasn't written (or synced) status.json yet —
        # freshness reference falls back to the run-dir mtime itself.
        liveness = _liveness(
            _mtime_or_none(run_dir), _launch_grace_seconds(), "launch_timeout", "launching"
        )
        return {"ok": True, "source": source, "started_at": None, **liveness}

    lane = None
    started_at = None
    stored_status = None
    response_path = None
    model = None
    provider = None
    provider_reason = None
    max_runtime_seconds = 0
    # Local (alpha-side) path — NEVER overridden from status.json's own
    # "transcript" field, which is host-relative (for worker4/Mac it's the
    # /Users/davidgomez/... path baked into run.sh, not alpha's synced-back
    # mirror) and would make os.path.getsize() below always fail/read 0.
    transcript = os.path.join(transcripts_dir, f"{token}.json")
    try:
        with open(status_json, "r", encoding="utf-8") as f:
            st = json.load(f)
        lane = st.get("lane")
        started_at = st.get("started_at")
        stored_status = st.get("status")
        response_path = st.get("response_path")
        model = st.get("model")
        provider = st.get("provider")
        provider_reason = st.get("provider_reason")
        max_runtime_seconds = st.get("max_runtime_seconds", 0)
    except Exception:
        pass

    if not os.path.exists(done_path):
        heartbeat_path = os.path.join(run_dir, "heartbeat")
        ref_epoch = _mtime_or_none(heartbeat_path)
        if ref_epoch is None:
            ref_epoch = _mtime_or_none(status_json)
        liveness = _liveness(
            ref_epoch, _stale_after_seconds(), "heartbeat_timeout", stored_status or "running"
        )
        # A status observation proves the run is still live and safely renews
        # the quota lease; this prevents long jobs from aging out while watched.
        _quota_router().renew(token, max(_stale_after_seconds() * 2, 900))
        return {
            "ok": True,
            "source": source,
            "started_at": started_at,
            "model": model,
            "provider": provider,
            "provider_reason": provider_reason,
            "max_runtime_seconds": max_runtime_seconds,
            **liveness,
        }

    try:
        with open(done_path, "r", encoding="utf-8") as f:
            exit_code = int((f.read().strip() or "-1"))
    except Exception:
        exit_code = -1

    if not response_path and lane in LANES:
        response_path = os.path.join(VAULT, from_seat, lane, "latest_response.md")

    try:
        transcript_bytes = os.path.getsize(transcript)
    except Exception:
        transcript_bytes = 0

    # Per-token slot first (zero contention — keyed by the exact token being
    # polled, so a newer run for the same seat+lane cannot clobber it before this
    # older run's status is read). Fall back to the shared latest_response.md for
    # runs launched before this per-token delivery path existed.
    token_response_path = os.path.join(run_dir, "response.md")
    response_token_match = False
    if os.path.isfile(token_response_path):
        response_token_match = _file_contains_token(token_response_path, token)
    elif response_path and os.path.isfile(response_path):
        response_token_match = _file_contains_token(response_path, token)

    terminal = None
    terminal_path = os.path.join(run_dir, "terminal.json")
    try:
        with open(terminal_path, "r", encoding="utf-8") as handle:
            candidate = json.load(handle)
        if isinstance(candidate, dict):
            terminal = candidate
    except Exception:
        pass

    result = {
        "ok": True,
        "status": "done",
        "source": source,
        "exit_code": exit_code,
        "transcript": transcript,
        "transcript_bytes": transcript_bytes,
        "response_token_match": response_token_match,
        "started_at": started_at,
        "model": model,
        "provider": provider,
        "provider_reason": provider_reason,
        "max_runtime_seconds": max_runtime_seconds,
    }
    if terminal is not None:
        result["terminal"] = terminal
        result["terminal_reason"] = terminal.get("reason")
    if exit_code == 0 and response_token_match:
        release_reason = "terminal_success"
    elif _terminal_provider_failure(run_dir, terminal):
        release_reason = "terminal_provider_failed"
    else:
        release_reason = "terminal_failed"
    _quota_router().release(token, release_reason)
    return result


def _node_status(token: str, node: str) -> dict:
    """Poll a run by token for one node.

    WORKER1-1: the node root is authoritative for new runs; the legacy seat root is
    consulted afterwards so historical records stay pollable by their token.
    """
    roots = {
        "delta": (
            (DELTA_RUNS, DELTA_TRANSCRIPTS, "from-delta", "delta"),
            (WORKER1_RUNS, WORKER1_TRANSCRIPTS, "from-worker1", "worker1"),
        ),
        "charlie": (
            (CHARLIE_RUNS, CHARLIE_TRANSCRIPTS, "from-charlie", "charlie"),
            (WORKER3_RUNS, WORKER3_TRANSCRIPTS, "from-worker3", "worker3"),
        ),
        "alpha": (
            (ALPHA_RUNS, ALPHA_TRANSCRIPTS, "from-alpha", "alpha"),
            (WORKER2_RUNS, WORKER2_TRANSCRIPTS, "from-worker2", "worker2"),
        ),
    }[node]
    result = None
    for runs, transcripts, from_dir, label in roots:
        result = _remote_status(token, runs, transcripts, from_dir, label)
        if result.get("status") != "missing":
            return result
    return result


def _delta_status(token: str) -> dict:
    return _node_status(token, "delta")


def worker1_status(token: str) -> dict:
    """Legacy alias retained for callers that still name the seat."""
    return _delta_status(token)


def _alpha_status(token: str) -> dict:
    return _node_status(token, "alpha")


def _worker3_status(token: str) -> dict:
    """Legacy alias; charlie is the node."""
    return _node_status(token, "charlie")


def _charlie_status(token: str) -> dict:
    return _node_status(token, "charlie")


def _localworker_status(token: str) -> dict:
    """Poll a kick_localworker run by token — vault-synced LOCALWORKER_RUNS, no /proc (remote pid)."""
    return _remote_status(token, LOCALWORKER_RUNS, LOCALWORKER_TRANSCRIPTS, "from-localworker", "localworker")


def _worker2_status(token: str) -> dict:
    """Legacy alias; alpha is the node."""
    return _node_status(token, "alpha")


def _worker4_status(token: str) -> dict:
    """Poll a kick_worker4 run by token."""
    return _remote_status(token, WORKER4_RUNS, WORKER4_TRANSCRIPTS, "from-worker4", "worker4")


def _run_entry(run_dir: str, token: str, source: str) -> dict:
    """Classify a single run dir as {token, status, started_at, pid, source, ...}.

    Both seats are remote to worker2, so there is NO pid liveness probe — we trust
    the `done` sentinel, the heartbeat file, and the run.sh-written status.json.
    status.json absent means the run was just launched and hasn't synced/landed
    yet ("launching", or "stale"/launch_timeout past TOWER_LAUNCH_GRACE_SECONDS
    since the run-dir mtime). A `done`-less run past TOWER_STALE_AFTER_SECONDS
    since its heartbeat/status.json mtime is "stale"/heartbeat_timeout — same
    classification _remote_status applies for a single-token poll. Factored out
    of _list_runs_in so relay_audit can reuse it without re-deriving thresholds.
    """
    started_at = None
    pid = None
    stored_status = None
    model = None
    status_json = os.path.join(run_dir, "status.json")
    has_status_json = os.path.exists(status_json)
    try:
        with open(status_json, "r", encoding="utf-8") as f:
            st = json.load(f)
        started_at = st.get("started_at")
        pid = st.get("pid")
        stored_status = st.get("status")
        model = st.get("model")
    except Exception:
        pass

    entry = {
        "token": token,
        "started_at": started_at,
        "pid": pid,
        "source": source,
        "model": model,
    }

    if os.path.exists(os.path.join(run_dir, "done")):
        entry["status"] = "done"
    elif not has_status_json:
        entry.update(_liveness(
            _mtime_or_none(run_dir), _launch_grace_seconds(), "launch_timeout", "launching"
        ))
    else:
        heartbeat_path = os.path.join(run_dir, "heartbeat")
        ref_epoch = _mtime_or_none(heartbeat_path)
        if ref_epoch is None:
            ref_epoch = _mtime_or_none(status_json)
        entry.update(_liveness(
            ref_epoch, _stale_after_seconds(), "heartbeat_timeout", stored_status or "launching"
        ))

    return entry


def _list_runs_in(root: str, source: str) -> list:
    """List one run-state root as compact {token, status, started_at, pid, source}.

    Missing root -> [] (forward-compat: a runs lane may not exist yet).
    """
    out = []
    if not os.path.isdir(root):
        return out
    for token in sorted(os.listdir(root)):
        run_dir = os.path.join(root, token)
        if not os.path.isdir(run_dir):
            continue
        out.append(_run_entry(run_dir, token, source))
    return out


def status(token=None) -> dict:
    """Poll relay runs. Never blocks; reads only the filesystem.

    With a token: try the Worker1 run dir first (worker1_status); if that's "missing", fall
    through to the Worker3 run dir (_worker3_status). Tokens are globally unique, so this
    transparently reports whichever seat ran it.

    Without a token: list ALL runs across Worker1 (WORKER1_RUNS) and Worker3 (WORKER3_RUNS — either
    dir may not exist yet, in which case it is skipped). Each entry is
    {token, status, started_at, pid, source}. Returns {ok:True, runs:[...]}.
    """
    if token:
        worker1 = worker1_status(token)
        if worker1.get("status") != "missing":
            return worker1
        worker3 = _worker3_status(token)
        if worker3.get("status") != "missing":
            return worker3
        localworker = _localworker_status(token)
        if localworker.get("status") != "missing":
            return localworker
        worker4 = _worker4_status(token)
        if worker4.get("status") != "missing":
            return worker4
        return _worker2_status(token)
    runs = (
        _list_runs_in(DELTA_RUNS, "delta")
        + _list_runs_in(CHARLIE_RUNS, "charlie")
        + _list_runs_in(ALPHA_RUNS, "alpha")
        # legacy seat roots keep historical runs listed
        + _list_runs_in(WORKER1_RUNS, "worker1")
        + _list_runs_in(WORKER3_RUNS, "worker3")
        + _list_runs_in(LOCALWORKER_RUNS, "localworker")
        + _list_runs_in(WORKER4_RUNS, "worker4")
        + _list_runs_in(WORKER2_RUNS, "worker2")
    )
    return {"ok": True, "runs": runs}


# --- Bounded run inspection --------------------------------------------------
_INSPECT_ALLOWED = frozenset(("status", "response", "transcript_tail", "tool_results"))
_INSPECT_DEFAULT = ("status", "response", "transcript_tail", "tool_results")
_INSPECT_MIN_BYTES = 1024
_INSPECT_DEFAULT_BYTES = 65536
_INSPECT_MAX_BYTES = 262144
_INSPECT_DEFAULT_EVENTS = 20
_INSPECT_MAX_EVENTS = 100


def _inspect_sources():
    """Seat roots used for exact-token discovery, evaluated fresh for tests."""
    return (
        ("delta", DELTA_RUNS, DELTA_TRANSCRIPTS, "from-delta"),
        ("charlie", CHARLIE_RUNS, CHARLIE_TRANSCRIPTS, "from-charlie"),
        ("alpha", ALPHA_RUNS, ALPHA_TRANSCRIPTS, "from-alpha"),
        # legacy seat roots, read-only: historical records keep their names
        ("worker1", WORKER1_RUNS, WORKER1_TRANSCRIPTS, "from-worker1"),
        ("worker3", WORKER3_RUNS, WORKER3_TRANSCRIPTS, "from-worker3"),
        ("localworker", LOCALWORKER_RUNS, LOCALWORKER_TRANSCRIPTS, "from-localworker"),
        ("worker4", WORKER4_RUNS, WORKER4_TRANSCRIPTS, "from-worker4"),
        ("worker2", WORKER2_RUNS, WORKER2_TRANSCRIPTS, "from-worker2"),
    )


def _inspect_clamp(value, default, floor, ceiling):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(floor, min(ceiling, parsed))


def _clip_utf8(value, limit):
    """Return (text, truncated, returned_utf8_bytes) under an exact byte cap."""
    if value is None:
        value = ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            value = repr(value)
    raw = value.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return value, False, len(raw)
    clipped = raw[: max(0, limit)].decode("utf-8", errors="ignore")
    return clipped, True, len(clipped.encode("utf-8"))


class _InspectTextBudget:
    def __init__(self, limit):
        self.limit = max(0, int(limit))
        self.used = 0

    @property
    def remaining(self):
        return max(0, self.limit - self.used)

    def take(self, value, requested=None):
        allowance = self.remaining
        if requested is not None:
            allowance = min(allowance, max(0, int(requested)))
        text, truncated, used = _clip_utf8(value, allowance)
        self.used += used
        return text, truncated


def _read_head_tail(path, budget):
    """Read a bounded response while retaining both VERDICT head and token tail."""
    total = os.path.getsize(path)
    if total <= budget:
        with open(path, "rb") as handle:
            raw = handle.read(budget)
        text = raw.decode("utf-8", errors="replace")
        returned_text_bytes = len(text.encode("utf-8"))
        return text, False, len(raw), [[0, len(raw)]], returned_text_bytes
    marker = b"\n\n[... bounded middle omitted ...]\n\n"
    source_budget = max(0, budget - len(marker))
    head_bytes = source_budget // 2
    tail_bytes = source_budget - head_bytes
    with open(path, "rb") as handle:
        head = handle.read(head_bytes)
        handle.seek(max(0, total - tail_bytes))
        tail = handle.read(tail_bytes)
    text = (head + marker + tail).decode("utf-8", errors="replace")
    returned_text_bytes = len(text.encode("utf-8"))
    return text, True, len(head) + len(tail), [[0, len(head)], [total - len(tail), total]], returned_text_bytes


def _file_contains_token(path, token):
    """Streaming exact-token search: whole-file proof with bounded memory."""
    needle = token.encode("utf-8")
    overlap = max(0, len(needle) - 1)
    carry = b""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    return False
                blob = carry + chunk
                if needle in blob:
                    return True
                carry = blob[-overlap:] if overlap else b""
    except OSError:
        return False


def _read_jsonl_tail(path, scan_bytes, max_events):
    total = os.path.getsize(path)
    start = max(0, total - scan_bytes)
    with open(path, "rb") as handle:
        handle.seek(start)
        raw = handle.read(scan_bytes)
    dropped_partial = False
    if start:
        newline = raw.find(b"\n")
        if newline < 0:
            raw = b""
        else:
            raw = raw[newline + 1:]
        dropped_partial = True
    parsed = []
    parse_errors = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            parse_errors += 1
            continue
        if isinstance(event, dict):
            parsed.append(event)
    if len(parsed) > max_events:
        parsed = parsed[-max_events:]
    return parsed, {
        "path": path,
        "total_bytes": total,
        "scanned_bytes": len(raw),
        "scan_start": start,
        "truncated": start > 0,
        "dropped_partial_first_line": dropped_partial,
        "parse_errors": parse_errors,
        "returned_event_count": len(parsed),
    }


def _event_blocks(event):
    message = event.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), list):
        return message["content"]
    if isinstance(event.get("content"), list):
        return event["content"]
    return []


def _summarize_events(events, budget):
    summaries = []
    for event in events:
        summary = {"type": event.get("type")}
        for key in ("subtype", "timestamp", "error", "terminal_reason"):
            if event.get(key) is not None:
                summary[key] = event.get(key)
        message = event.get("message")
        if isinstance(message, dict):
            for key in ("role", "model", "stop_reason"):
                if message.get(key) is not None:
                    summary[key] = message.get(key)
        blocks = []
        for block in _event_blocks(event):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            compact = {"type": kind}
            if kind == "tool_use":
                compact.update({"id": block.get("id"), "name": block.get("name")})
                text, clipped = budget.take(block.get("input"), requested=2048)
                compact["input"] = text
                compact["input_truncated"] = clipped
            elif kind == "tool_result":
                compact.update({
                    "tool_use_id": block.get("tool_use_id"),
                    "is_error": bool(block.get("is_error", False)),
                })
                text, clipped = budget.take(block.get("content"), requested=4096)
                compact["content"] = text
                compact["content_truncated"] = clipped
            elif kind == "text":
                text, clipped = budget.take(block.get("text"), requested=2048)
                compact["text"] = text
                compact["text_truncated"] = clipped
            else:
                continue
            blocks.append(compact)
        if blocks:
            summary["content"] = blocks
        if event.get("result") is not None:
            text, clipped = budget.take(event.get("result"), requested=2048)
            summary["result"] = text
            summary["result_truncated"] = clipped
        summaries.append(summary)
    return summaries


def _select_tool_results(events, budget, max_events):
    names = {}
    selected = []
    for event in events:
        for block in _event_blocks(event):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id"):
                names[block["id"]] = block.get("name")
            elif block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                content, clipped = budget.take(block.get("content"), requested=8192)
                selected.append({
                    "tool_use_id": tool_id,
                    "tool_name": names.get(tool_id),
                    "is_error": bool(block.get("is_error", False)),
                    "content": content,
                    "content_truncated": clipped,
                })
    return selected[-max_events:]


def inspect_run(token, include=None, max_events=20, max_bytes=65536):
    """Return bounded, read-only status and correctness evidence for one run."""
    token = (token or "").strip()
    if not token:
        return {"ok": False, "error": "empty token"}
    if not _LAUNCH_TOKEN_RE.fullmatch(token):
        return {"ok": False, "error": "malformed token"}
    requested = list(_INSPECT_DEFAULT if include is None else include)
    if not requested or any(item not in _INSPECT_ALLOWED for item in requested):
        return {"ok": False, "error": "bad include", "allowed": sorted(_INSPECT_ALLOWED)}
    requested = list(dict.fromkeys(requested))
    event_limit = _inspect_clamp(max_events, _INSPECT_DEFAULT_EVENTS, 1, _INSPECT_MAX_EVENTS)
    byte_limit = _inspect_clamp(
        max_bytes, _INSPECT_DEFAULT_BYTES, _INSPECT_MIN_BYTES, _INSPECT_MAX_BYTES
    )

    matches = []
    for seat, runs_root, transcripts_dir, from_seat in _inspect_sources():
        run_dir = os.path.join(runs_root, token)
        if os.path.isdir(run_dir):
            matches.append((seat, runs_root, transcripts_dir, from_seat, run_dir))
    if not matches:
        return {"ok": False, "error": "run not found", "token": token}
    if len(matches) > 1:
        return {"ok": False, "error": "ambiguous token", "token": token,
                "seats": [match[0] for match in matches]}

    seat, runs_root, transcripts_dir, from_seat, run_dir = matches[0]
    result = {
        "ok": True,
        "token": token,
        "seat": seat,
        "included": requested,
        "limits": {
            "max_events": event_limit,
            "max_output_text_bytes": byte_limit,
            "max_transcript_scan_bytes": byte_limit,
        },
    }
    section_names = [name for name in ("response", "transcript_tail", "tool_results") if name in requested]
    base = byte_limit // max(1, len(section_names))
    remainders = byte_limit - base * len(section_names)
    budgets = {}
    for index, name in enumerate(section_names):
        budgets[name] = _InspectTextBudget(base + (1 if index < remainders else 0))

    if "status" in requested:
        result["status"] = _remote_status(token, runs_root, transcripts_dir, from_seat, seat)

    if "response" in requested:
        response_path = os.path.join(run_dir, "response.md")
        response_source = "per_token"
        durable = True
        if not os.path.isfile(response_path):
            lane = None
            try:
                with open(os.path.join(run_dir, "status.json"), encoding="utf-8") as handle:
                    lane = json.load(handle).get("lane")
            except Exception:
                pass
            if lane in LANES:
                response_path = os.path.join(VAULT, from_seat, lane, "latest_response.md")
                response_source = "shared_legacy_fallback"
                durable = False
        if os.path.isfile(response_path):
            text, truncated, source_bytes, ranges, returned_text_bytes = _read_head_tail(
                response_path, budgets["response"].limit
            )
            budgets["response"].used = min(budgets["response"].limit, returned_text_bytes)
            result["response"] = {
                "available": True,
                "source": response_source,
                "durable_per_token": durable,
                "path": response_path,
                "total_bytes": os.path.getsize(response_path),
                "returned_source_bytes": source_bytes,
                "returned_text_bytes": returned_text_bytes,
                "returned_ranges": ranges,
                "truncated": truncated,
                "token_match": _file_contains_token(response_path, token),
                "text": text,
            }
        else:
            result["response"] = {"available": False, "token_match": False}

    transcript_path = os.path.join(transcripts_dir, f"{token}.json")
    events = []
    transcript_meta = {
        "path": transcript_path,
        "available": os.path.isfile(transcript_path),
        "total_bytes": os.path.getsize(transcript_path) if os.path.isfile(transcript_path) else 0,
    }
    if os.path.isfile(transcript_path) and (
        "transcript_tail" in requested or "tool_results" in requested
    ):
        events, transcript_meta = _read_jsonl_tail(transcript_path, byte_limit, event_limit)
        transcript_meta["available"] = True

    if "transcript_tail" in requested:
        result["transcript_tail"] = dict(transcript_meta)
        result["transcript_tail"]["events"] = _summarize_events(
            events, budgets["transcript_tail"]
        ) if events else []
        result["transcript_tail"]["returned_text_bytes"] = budgets["transcript_tail"].used

    if "tool_results" in requested:
        selected = _select_tool_results(events, budgets["tool_results"], event_limit) if events else []
        result["tool_results"] = {
            "transcript_available": bool(transcript_meta.get("available")),
            "selected_count": len(selected),
            "results": selected,
            "returned_text_bytes": budgets["tool_results"].used,
            "selection_scope": "parsed transcript tail",
            "selection_complete": not bool(transcript_meta.get("truncated", False)),
            "transcript_total_bytes": transcript_meta.get("total_bytes", 0),
            "transcript_scanned_bytes": transcript_meta.get("scanned_bytes", 0),
        }

    result["output_text_bytes"] = sum(budget.used for budget in budgets.values())
    result["output_text_truncated"] = result["output_text_bytes"] >= byte_limit
    return result


# --- Relay audit --------------------------------------------------------------
# Read-only fleet-health snapshot across all five canonical seats. Distinct from
# status(): status() answers "what happened to token X" / "list all runs";
# relay_audit() answers "is the relay itself healthy" — per-seat root presence,
# status-bucket counts, and the single most-recent run, plus the Syncthing
# transport dependency the whole relay rides on. Never mutates anything.

def _audit_seat_roots():
    """(seat, runs_root, from_seat) for the five canonical seats, read fresh.

    A function rather than a module-level tuple so it re-reads the current
    WORKER1_RUNS/WORKER3_RUNS/... globals on every call (same pattern status() relies
    on) instead of freezing their import-time values — the run roots are
    monkeypatched in tests, and WORKER2_RUNS is env-tunable.
    """
    return (
        ("delta", DELTA_RUNS, "from-delta"),
        ("charlie", CHARLIE_RUNS, "from-charlie"),
        ("alpha", ALPHA_RUNS, "from-alpha"),
        # legacy seat roots, read-only
        ("worker1", WORKER1_RUNS, "from-worker1"),
        ("worker3", WORKER3_RUNS, "from-worker3"),
        ("localworker", LOCALWORKER_RUNS, "from-localworker"),
        ("worker4", WORKER4_RUNS, "from-worker4"),
        ("worker2", WORKER2_RUNS, "from-worker2"),
    )

_DEFAULT_AUDIT_MAX_RECENT = 20
_AUDIT_MAX_RECENT_FLOOR = 1
_AUDIT_MAX_RECENT_CEILING = 100

# homelab-vault is one vault among several under ~/Vaults, dedicated to
# relay/operational state (to-<seat>/from-<seat> lanes, runs, transcripts,
# sessions, audits) — Loupe project knowledge lives in the separate loupe-vault.
# vault_root is the broader Vaults root, relay_root is homelab-vault (aka
# VAULT above) specifically; reported separately since vault_root spans
# siblings the relay doesn't own.
_VAULTS_ROOT = os.path.join(HOME, "Vaults")


def _clamp_max_recent(max_recent) -> int:
    try:
        val = int(max_recent)
    except (TypeError, ValueError):
        val = _DEFAULT_AUDIT_MAX_RECENT
    return max(_AUDIT_MAX_RECENT_FLOOR, min(_AUDIT_MAX_RECENT_CEILING, val))


def _syncthing_service_state() -> str:
    """Read-only `systemctl --user is-active syncthing.service`, never raises.

    Syncthing runs as a per-user unit on this host, so the probe must query
    the user service manager (`--user`) rather than the system one — the
    system-level unit is inactive by design and would otherwise read as a
    false alarm. Returns "active"/"inactive"/whatever systemctl reports, or
    "unknown" if the check itself fails (binary missing, timeout, non-systemd
    host) — a failed probe must not fail the whole audit.
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "syncthing.service"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        state = (result.stdout or "").strip()
        return state or "unknown"
    except Exception:
        return "unknown"


def _seat_audit(seat: str, runs_root: str, from_seat: str, max_recent: int) -> dict:
    """Bounded, read-only health snapshot for one seat's run root.

    Only stats/classifies the `max_recent` most-recently-modified run dirs (by
    run-dir mtime) — total_run_count is a cheap directory count over the whole
    root, but per-run classification (which opens status.json/heartbeat/done)
    is bounded to that window so a root with a long run history can't turn an
    audit call into an unbounded filesystem walk.
    """
    from_root = os.path.join(VAULT, from_seat)
    audit = {
        "seat": seat,
        "run_root": runs_root,
        "run_root_exists": os.path.isdir(runs_root),
        "from_seat_root": from_root,
        "from_seat_root_exists": os.path.isdir(from_root),
        "total_run_count": 0,
        "inspected_run_count": 0,
        "inspection_limit": max_recent,
        "status_counts_scope": "inspected_recent_runs_only",
        "status_counts": {"done": 0, "running": 0, "launching": 0, "stale": 0},
        "stale_count": 0,
        "most_recent_run": None,
        "most_recent_done_at": None,
    }
    if not audit["run_root_exists"]:
        return audit

    candidates = []
    for token in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, token)
        if os.path.isdir(run_dir):
            candidates.append((_mtime_or_none(run_dir) or 0.0, token, run_dir))

    audit["total_run_count"] = len(candidates)
    if not candidates:
        return audit

    candidates.sort(key=lambda c: c[0], reverse=True)
    recent = candidates[:max_recent]
    audit["inspected_run_count"] = len(recent)

    most_recent_done_epoch = None
    for _mtime, token, run_dir in recent:
        entry = _run_entry(run_dir, token, seat)
        st = entry.get("status", "unknown")
        audit["status_counts"][st] = audit["status_counts"].get(st, 0) + 1
        if st == "stale":
            audit["stale_count"] += 1
        if st == "done":
            done_epoch = _mtime_or_none(os.path.join(run_dir, "done"))
            if done_epoch is not None and (
                most_recent_done_epoch is None or done_epoch > most_recent_done_epoch
            ):
                most_recent_done_epoch = done_epoch

    top_epoch, top_token, top_run_dir = recent[0]
    top_entry = _run_entry(top_run_dir, top_token, seat)
    most_recent = {
        "token": top_entry.get("token"),
        "status": top_entry.get("status"),
        "started_at": top_entry.get("started_at"),
        "model": top_entry.get("model"),
    }
    for optional_field in ("age_seconds", "last_seen_at", "stale_reason"):
        if optional_field in top_entry:
            most_recent[optional_field] = top_entry[optional_field]
    audit["most_recent_run"] = most_recent

    if most_recent_done_epoch is not None:
        audit["most_recent_done_at"] = _iso_from_epoch(most_recent_done_epoch)

    return audit


def relay_audit(max_recent: int = 20) -> dict:
    """Read-only fleet-health snapshot across worker1/worker3/localworker/worker4/worker2.

    Never mutates runs, prompts, notes, or services — filesystem stats/reads and
    one bounded `systemctl is-active` subprocess call only. `max_recent` (clamped
    to 1..100) bounds how many of each seat's most-recently-modified run dirs get
    opened/classified; `total_run_count` is still the true total in the root.
    Reuses the same stale/launching classification `_run_entry`/`_liveness` use
    for status() — no duplicated threshold logic.
    """
    clamped = _clamp_max_recent(max_recent)
    seats = [
        _seat_audit(seat, runs_root, from_seat, clamped)
        for seat, runs_root, from_seat in _audit_seat_roots()
    ]
    return {
        "ok": True,
        "generated_at": _now(),
        "max_recent": clamped,
        "scope": {
            "total_run_count": "all_run_directories",
            "status_counts": "inspected_recent_runs_only",
            "selection_order": "run_directory_mtime_descending",
        },
        "sync_dependency": "syncthing",
        "syncthing_service": _syncthing_service_state(),
        "vault_root": _VAULTS_ROOT,
        "vault_root_exists": os.path.isdir(_VAULTS_ROOT),
        "relay_root": VAULT,
        "relay_root_exists": os.path.isdir(VAULT),
        "seats": seats,
    }

"""Vault-search MCP server (Streamable HTTP).

Reads the transported sqlite-vec index, CPU query-embeds with the pinned nomic
recipe (see vaultsearch.py), serves vault notes, and (Access-gated) writes them.
Bound to 127.0.0.1:8765 — the Cloudflare tunnel is same-box on worker2, so the
origin only ever needs loopback.

Auth: a Cloudflare-Access JWT gate (CfAccessJWTMiddleware) validates the
`Cf-Access-Jwt-Assertion` header Cloudflare Access injects at the edge on EVERY
request. Missing/invalid -> 403. This closes the direct-LAN gap (anything that
reaches the origin without a valid Cloudflare-signed assertion is rejected).
"""
import base64
import fcntl
import ipaddress
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import tempfile
import textwrap
from contextlib import contextmanager
from datetime import datetime, timezone

import anyio
import jwt
from jwt import PyJWKClient
from mcp.server.transport_security import TransportSecuritySettings
from mcp.server.fastmcp import FastMCP
from mcp.types import Icon
from starlette.requests import Request
from starlette.responses import JSONResponse

import kicker
import vaultsearch as vs

log = logging.getLogger("tower.gate")
# Own non-truncating handler so the reject REASON CLASS is never clipped by the
# Rich handler another lib installs on the root logger. propagate=False keeps
# our line from being re-rendered (and truncated) by that handler.
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

# --- Cloudflare Access config (validate the edge-injected assertion) ---------
# Audiences live in the TOWER_CF_AUD env var (comma-separated), loaded from the
# user-owned EnvironmentFile /home/david/.config/tower/env (mode 600) via
# the tower.service unit — never hardcoded here so the values stay out of
# source control.


def _load_cf_aud() -> list:
    raw = (os.environ.get("TOWER_CF_AUD")
           or os.environ.get("FLEET_CF_AUD", ""))
    auds = [a.strip() for a in raw.split(",") if a.strip()]
    if not auds:
        log.warning(
            "cf-access gate: TOWER_CF_AUD is missing/empty -> "
            "external JWT validation will fail closed (all non-loopback requests rejected)"
        )
    return auds


CF_AUD = _load_cf_aud()
CF_TEAM = "https://davidgomez.cloudflareaccess.com"
CF_JWKS = "https://davidgomez.cloudflareaccess.com/cdn-cgi/access/certs"

# --- Trust channels (models breadcrumbs/history/server/requestTrust.js) --------
# NOTE: this previously said it mirrored fleet-dashboard/app/access.py. That file
# was deleted 2026-07-20 (4c3fceb) and its cookie-auth successor 2026-08-02
# (bdbcb16); Panel now has no in-process auth at all. The live precedent for
# "trust the tailnet, gate the tunnel" is breadcrumbs' requestTrust.js, which this
# follows: named channels, each requiring TWO facts, never one.
#
# cloudflared runs same-box on alpha and dials the origin at 127.0.0.1, so a
# tunnel-forwarded request ALSO has peer_ip == 127.0.0.1 — peer-IP alone is NOT a
# safe discriminator. cloudflared always injects these edge headers on origin
# requests; a genuine on-box process sends neither. So exempt loopback ONLY when
# BOTH headers are absent.
#
# The same two-fact rule extends the exemption to the tailnet (see
# TOWER_BIND_ADDRESSES below): a WireGuard-authenticated 100.64/10 peer
# with no edge headers is one of David's own devices reaching Tower directly. A
# tailnet peer that DOES carry edge headers is not trusted implicitly — it falls
# through to full JWT validation, so a forged cf-ray cannot downgrade a request
# into the trusted branch, and a spoofed source address cannot either (the reply
# would have to route back over the tailnet's authenticated tunnel).
#
# This is a deliberate, documented widening of what used to be loopback-only.
# It is safe only while Tower binds loopback + tailnet addresses EXPLICITLY and
# never 0.0.0.0 — a LAN-reachable bind would make the LAN implicitly trusted.
CF_EDGE_HEADERS = ("cf-connecting-ip", "cf-ray")
# Tailscale CGNAT range (v4) and the tailnet's ULA prefix (v6).
TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
TAILNET_V6_PREFIX = "fd7a:115c:a1e0:"


def normalize_peer(address):
    """Lowercase, strip an IPv6 %zone, and unwrap ::ffff: v4-mapped addresses.

    Without the unwrap, a dual-stack listener reports loopback as
    '::ffff:127.0.0.1', which the old exact-string comparison missed — the peer
    would have been sent down the JWT path and rejected. Ported from
    breadcrumbs' normalizePeerAddress for exactly that reason."""
    normalized = str(address or "").strip().lower()
    zone = normalized.find("%")
    if zone != -1:
        normalized = normalized[:zone]
    if normalized.startswith("::ffff:"):
        normalized = normalized[7:]
    return normalized


def _parsed_peer(address):
    try:
        return ipaddress.ip_address(normalize_peer(address))
    except ValueError:
        return None


def is_loopback_peer(address):
    """True for the whole 127/8 block and ::1 — not just the literal 127.0.0.1."""
    ip = _parsed_peer(address)
    return ip is not None and ip.is_loopback


def is_tailnet_peer(address):
    """True for a Tailscale peer: 100.64/10 (v4) or this tailnet's ULA (v6).

    Deliberately NARROWER than RFC1918 — the LAN is not trusted, only the
    WireGuard-authenticated tailnet. Matches breadcrumbs; Panel's ingest check is
    looser (all is_private) and is not the model followed here."""
    normalized = normalize_peer(address)
    ip = _parsed_peer(normalized)
    if ip is None:
        return False
    if ip.version == 6:
        return normalized.startswith(TAILNET_V6_PREFIX)
    return ip in TAILNET_V4


# --- Bind addresses -----------------------------------------------------------
# Default is loopback-only — byte-for-byte the historical behavior — so the
# tailnet path is opt-in from the unit and deleting the drop-in is a COMPLETE
# rollback with no source change. Addresses are bound EXPLICITLY, one socket
# each, never 0.0.0.0: binding the wildcard would make the LAN reachable and
# therefore implicitly trusted by nothing more than an oversight. Mirrors
# breadcrumbs' BREADCRUMBS_BIND_ADDRESSES.
TOWER_PORT = 8765
TAILNET_DNS_SUFFIX = "tail3327f9.ts.net"
TOWER_NODE = (os.environ.get("TOWER_NODE")
              or os.environ.get("FLEET_TOWER_NODE", "")).strip() or "alpha"


def _bind_addresses():
    raw = (os.environ.get("TOWER_BIND_ADDRESSES")
           or os.environ.get("FLEET_TOWER_BIND_ADDRESSES", "")).strip()
    addrs = [a.strip() for a in raw.split(",") if a.strip()]
    return addrs or ["127.0.0.1"]


BIND_ADDRESSES = _bind_addresses()


def classify_peer_trust(peer_ip, headers):
    """Name the channel a request arrived on: 'loopback', 'tailnet', or None.

    None means "not implicitly trusted" — the caller must fall through to full
    Cloudflare Access JWT validation. Returning a NAME rather than a bool is the
    point: the channel is logged, so which door a request came in by is visible
    in journalctl rather than inferred."""
    if any(h in headers for h in CF_EDGE_HEADERS):
        return None                      # came via the edge — JWT decides, always
    if is_loopback_peer(peer_ip):
        return "loopback"
    if is_tailnet_peer(peer_ip):
        return "tailnet"
    return None


class CfAccessJWTMiddleware:
    """Pure-ASGI gate: validate the Cloudflare Access JWT on every HTTP request.

    Pure ASGI (not BaseHTTPMiddleware) so it does not buffer the Streamable-HTTP
    response stream. The PyJWKClient is constructed ONCE and caches/refreshes the
    signing keys (lifespan default 300s).
    """

    def __init__(self, app):
        self.app = app
        self._jwks = PyJWKClient(CF_JWKS)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        # Implicitly trusted channels: genuine on-box loopback (so the dashboard
        # reuses the resident model for semantic search) and a direct tailnet peer.
        # Anything arriving through the edge — including from those same addresses —
        # carries cf-ray/cf-connecting-ip and falls through to full JWT validation
        # below, so the public surface stays fail-closed.
        client = scope.get("client")
        peer_ip = client[0] if client else None
        channel = classify_peer_trust(peer_ip, headers)
        if channel:
            log.info("cf-access gate: %s peer (no cf headers) -> exempt", channel)
            await self.app(scope, receive, send)
            return
        token = headers.get("cf-access-jwt-assertion")
        if not token:
            log.info("cf-access gate: assertion=absent -> reject (no-header)")
            await self._deny(send)
            return
        try:
            await anyio.to_thread.run_sync(self._verify, token)
        except Exception as e:
            # reason CLASS only, never token/claim VALUES
            log.info(
                "cf-access gate: assertion=present -> reject (%s)", type(e).__name__
            )
            await self._deny(send)
            return
        log.info("cf-access gate: assertion=present -> accept")
        await self.app(scope, receive, send)

    def _verify(self, token: str):
        if not CF_AUD:
            raise RuntimeError("cf-access: no audience configured (TOWER_CF_AUD unset)")
        signing_key = self._jwks.get_signing_key_from_jwt(token)
        jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=CF_AUD,
            issuer=CF_TEAM,
        )

    async def _deny(self, send):
        body = b'{"error":"access denied"}'
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tower-icon.png")


def _load_icons():
    try:
        with open(_ICON_PATH, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode("ascii")
        return [Icon(src=f"data:image/png;base64,{_b64}", mimeType="image/png", sizes=["64x64"])]
    except Exception as _e:
        print(f"[tower] icon not loaded ({_e}); serving without serverInfo.icons", flush=True)
        return None


_ICONS = _load_icons()


mcp = FastMCP(
    name="tower",
    instructions=(
        "Semantic search and read/write over David's Obsidian vaults (~/Vaults). "
        "Use search_vault(query, k) to find notes by meaning, get_note(path) to "
        "read a full note. Use search_history(query, k) only when canonical notes "
        "do not answer the question or exact prior conversation evidence matters; "
        "then use read_history(evidence_path, start_chunk, limit) for bounded context. "
        "Use write_note(path, content, mode) to create/update a .md "
        "note, and the Distill tools to condense or route work before it reaches a "
        "larger model. Paths are vault-relative (e.g. 'loupe-vault/HANDOFF.md'). "
        "Resolve the owning project first, then read its AGENTS.md and HANDOFF.md. "
        "The five fleet control-plane indexes live only at the homelab-vault root; "
        "journal decisions and learnings live under journal/inbox and derived monthly shards."
    ),
    icons=_ICONS,
    host="127.0.0.1",
    port=8765,
    streamable_http_path="/mcp",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            # WORKER1-5: tower.example.com is the post-rename origin. The old
            # hostname stays listed until the connector has been re-registered
            # and its DNS record retired, so the cutover has no gap.
            "tower.example.com",
            "tower.example.com",
            "standby-tower.example.com",
            "127.0.0.1:*",
            "localhost:*",
            # Tailnet entry points. DNS-rebinding protection runs INSIDE the MCP
            # transport — i.e. AFTER the auth gate — and answers a bare 421
            # "Invalid Host header" for anything not listed here. Without these a
            # trusted tailnet peer passes the gate and is then refused by the
            # transport, which is a genuinely confusing failure to diagnose.
            # Derived from the bind list so adding an address cannot desync them.
            f"{TOWER_NODE}:*",
            f"{TOWER_NODE}.{TAILNET_DNS_SUFFIX}:*",
            *[f"{a}:*" for a in BIND_ADDRESSES],
        ],
    ),
)


def tower_health_status() -> dict:
    """Return a bounded, non-secret readiness verdict for edge health probes.

    Process liveness is implied by reaching this function.  Readiness also
    requires the synchronized vault root, the canonical relay root, and the
    transported semantic index to be readable.  Provider quota or remote worker
    health is deliberately excluded: those are routed capabilities, not reasons
    to steer the entire Tower endpoint away from an otherwise healthy node.
    """
    vaults_root_ok = os.path.isdir(vs.VAULTS_ROOT) and os.access(
        vs.VAULTS_ROOT, os.R_OK | os.X_OK
    )
    relay_root_ok = os.path.isdir(kicker.VAULT) and os.access(
        kicker.VAULT, os.R_OK | os.X_OK
    )
    try:
        index = vs.index_metadata()
        index_ok = bool(index.get("ok"))
    except Exception as exc:  # fail closed without leaking path or exception text
        log.warning("healthz: semantic index check failed (%s)", type(exc).__name__)
        index_ok = False

    checks = {
        "vaults_root": bool(vaults_root_ok),
        "relay_root": bool(relay_root_ok),
        "semantic_index": bool(index_ok),
    }
    degraded_standby = (
        (os.environ.get("TOWER_STANDBY_DEGRADED")
         or os.environ.get("FLEET_STANDBY_DEGRADED", "")).strip() == "1"
    )
    required_checks = ("vaults_root", "relay_root")
    ready = (
        all(checks[name] for name in required_checks)
        if degraded_standby
        else all(checks.values())
    )
    return {
        "ok": ready,
        "service": "tower",
        "mode": "degraded-cold-standby" if degraded_standby else "primary",
        "checks": checks,
        "degraded_capabilities": (
            ["search_vault", "search_history", "index_metadata"]
            if degraded_standby and not index_ok
            else []
        ),
    }


@mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
async def healthz(_request: Request) -> JSONResponse:
    """Deep readiness route for authenticated Cloudflare health monitors."""
    status = tower_health_status()
    return JSONResponse(status, status_code=200 if status["ok"] else 503)


@mcp.tool()
def search_vault(query: str, k: int = 8) -> list[dict]:
    """Semantic (vector) search over the vault index.

    IMPORTANT QUOTA RULE: Before calling this tool across any vault or project, you MUST evaluate whether this task can be delegated to **Localworker** (`ask_localworker`/`summon_localworker`, local Ollama on charlie — zero cloud quota) to preserve cloud quota. Default to Localworker for routine recon, vault sweeps, and log scraping.

    Args:
        query: Natural-language search query.
        k: Number of top matches to return (default 8).

    Returns:
        List of {path, vault, title, score, excerpt}, best cosine match first.
        `score` is cosine similarity in [-1, 1] (higher = closer).
    """
    return vs.search(query, k=k)


@mcp.tool()
def index_metadata() -> dict:
    """Report semantic-index freshness and size without embedding a query.

    Returns exact document/chunk counts, database size/schema version, the
    database publication time explicitly sourced from file mtime, and the
    oldest/latest per-document indexed_at values. The current schema has no
    authoritative build timestamp, which is reported honestly rather than
    inferred from a document timestamp.
    """
    return vs.index_metadata()


@mcp.tool()
def search_history(
    query: str,
    k: int = 8,
    vault: str = "",
    surface: str = "",
    host: str = "",
) -> list[dict]:
    """Search exact historical conversation evidence semantically.

    Use this secondary corpus after `search_vault` when canonical session notes,
    decisions, and handoffs do not contain enough detail, or when the exact prior
    wording/tool evidence matters. Results are redacted bounded excerpts with
    pointers to the manifest, raw transcript, and primary conversation card.

    Args:
        query: Natural-language description of the prior work or evidence.
        k: Top matches, clamped to 1..50.
        vault: Optional exact owning-vault filter, such as homelab-vault.
        surface: Optional exact surface filter, such as codex or claude-code.
        host: Optional exact fleet-host filter.
    """
    return vs.search_history(query, k=k, vault=vault, surface=surface, host=host)


@mcp.tool()
def read_history(
    evidence_path: str,
    start_chunk: int = 0,
    limit: int = 6,
    max_chars: int = 16000,
) -> dict:
    """Read bounded neighboring chunks from one transcript-evidence result.

    Pass the exact `evidence_path` returned by `search_history`. This reads the
    normalized redacted evidence projection, never the full raw archive. Follow
    the returned `next_chunk` for another bounded page only when necessary.
    """
    return vs.read_history(
        evidence_path,
        start_chunk=start_chunk,
        limit=limit,
        max_chars=max_chars,
    )


@mcp.tool()
def get_note(path: str) -> str:
    """Read the full text of a vault note.

    IMPORTANT QUOTA RULE: Before calling this tool across any vault or project, you MUST evaluate whether this task can be delegated to **Localworker** (`ask_localworker`/`summon_localworker`, local Ollama on charlie — zero cloud quota) to preserve cloud quota. Default to Localworker for routine recon, vault sweeps, and log scraping.

    Args:
        path: Vault-relative path under ~/Vaults (e.g. 'loupe-vault/HANDOFF.md').

    Returns:
        The full UTF-8 content of the note.
    """
    return vs.read_note(path)


@mcp.tool()
def read_vault_range(
    path: str,
    offset: int = 0,
    limit: int = 16384,
    unit: str = "bytes",
    max_bytes: int = 65536,
    expected_sha256: str = "",
    expected_mtime_ns: int = 0,
) -> dict:
    """Read one bounded, drift-detecting page of a vault note or transcript.

    Paths are vault-relative and limited to UTF-8 Markdown/JSON. Use byte mode
    for exact transcript offsets and line mode for complete note lines. Reuse
    the returned sha256 and mtime_ns as expectations on later pages so a file
    change cannot silently mix two versions.
    """
    return vs.read_vault_range(
        path, offset, limit, unit, max_bytes, expected_sha256, expected_mtime_ns
    )


@mcp.tool()
def write_note(path: str, content: str, mode: str = "overwrite") -> dict:
    """Create or update a .md note under ~/Vaults.

    Args:
        path: Vault-relative path to a .md file (e.g. 'loupe-vault/note.md').
            Absolute paths, '..' components, and paths outside ~/Vaults are rejected.
            homelab-vault/to-*/from-* (relay machinery) is also rejected —
            use stage_prompt() to stage an outbound prompt; inbound responses
            are written by the script-owned delivery path, not this tool.
        content: UTF-8 text to write.
        mode: 'overwrite' (default), 'append', or 'prepend'.

    Returns:
        {ok: true, path, bytes, mode} on success, else {ok: false, error}.
    """
    return vs.write_note(path, content, mode=mode)


@mcp.tool()
def append_journal_entry(vault: str, kind: str, seat: str, stamp: str, text: str) -> dict:
    """Append one validated bullet to a daily journal inbox.

    The tool derives the only allowed destination from vault, kind, seat, and
    the verified UTC/Central stamp. Raw write_note access to journal inboxes is
    blocked so callers cannot accidentally overwrite an existing daily file.
    """
    return vs.append_journal_entry(vault, kind, seat, stamp, text)


@mcp.tool()
def stage_prompt(
    seat: str, lane: str, token: str, content: str, archive_previous: bool = True
) -> dict:
    """Compatibility-only staging for a validated homelab-vault relay prompt.

    New work should use atomic dispatch(). This tool writes
    homelab-vault/to-<seat>/<lane>/latest.md. Raw write_note is
    blocked for that destination — use this tool instead so every staged
    prompt gets seat/lane/token validation and (by default) the previous
    latest.md is archived before it's replaced.

    Args:
        seat: One of 'worker1', 'worker3', 'localworker', 'worker2', or retired compatibility seat 'worker4'.
        lane: One of 'prompts' (BUILD/WORK) or 'recon' (read-only investigation).
        token: The relay token (e.g. 'FLEET-WORKER2-BUILD-20260710-slug' or
            legacy 'FLEET-BUILD-20260710-slug'). Must appear verbatim in content.
        content: Full Markdown content of the prompt. This is a complete
            replacement of latest.md, never an append/prepend.
        archive_previous: If true (default), copy the existing latest.md to
            homelab-vault/to-<seat>/<lane>/archive/<UTC ts>-latest.md
            before replacing it.

    Returns:
        {ok: true, path, seat, lane, token, bytes, archive_path?} on success,
        else {ok: false, error} (content is never echoed in an error).
    """
    return vs.stage_prompt(seat, lane, token, content, archive_previous=archive_previous)


@mcp.tool()
def recommend_model(
    lane: str = "worker",
    task_size: str = "small",
    current_provider: str = "",
    localworker_eligible: bool = False,
    allowed_providers: str = "claude,codex,gemini",
) -> dict:
    """Recommend a quota-aware model without launching or reserving it.

    Args:
        lane: worker. The parameter remains explicit for old clients, but
            strategy routing is no longer supported.
        task_size: tiny, small, medium, or large.
        current_provider: Optional current provider for the eight-point
            anti-flapping rule.
        localworker_eligible: True only when the task fits Localworker's exact
            zero-cloud worker contract.
        allowed_providers: Comma-separated subset of claude,codex,gemini.

    Returns:
        The winning provider/model/state/score plus every candidate's reserve,
        reset, confidence, and reason. This call never launches work.
    """
    allowed = tuple(
        provider.strip().lower()
        for provider in allowed_providers.split(",")
        if provider.strip()
    )
    return kicker._quota_router().recommend(
        lane=lane,
        size=task_size,
        current_provider=current_provider or None,
        localworker_eligible=localworker_eligible,
        allowed_providers=allowed,
    )


@mcp.tool()
def model_routing_status() -> dict:
    """Return the current worker recommendation and auditable Router v2 state."""
    router = kicker._quota_router()
    diagnostics = router.diagnostics()
    return {
        "ok": True,
        "worker": router.recommend(lane="worker", size="small"),
        # Retain the v1 top-level field for existing Panel/strategy clients.
        "active_reservations": diagnostics["active_reservations"],
        "router": diagnostics,
    }


_CLOUD_WORKER_NODES = ("delta", "charlie", "alpha")
_CLOUD_WORKER_SEATS = _CLOUD_WORKER_NODES  # legacy alias, removed with the seat vocabulary
# WORKER1-1: a node is its own home. Retained as an identity map so call sites that
# still ask for a worker's home host keep working during the migration.
_WORKER_HOME_HOST = {n: n for n in _CLOUD_WORKER_NODES}
# WORKER1-2: both the pre-rename and post-rename node names are accepted while the
# hosts are being renamed; canonical_node() collapses them.
_TARGET_HOSTS = frozenset(
    ("macbook", "alpha", "charlie", "delta", "echo", "bravo",
     "alpha", "delta", "charlie")
)
_TARGET_SCOPE_RE = re.compile(r"^(?:repo|service|host|path):[A-Za-z0-9_./:@+-]{1,240}$")
_TERMINAL_RUN_STATES = frozenset(("done", "stale"))
_WORKER_ROUTER_LOCK = os.path.join(
    os.path.expanduser("~"), ".local", "state", "fleet", "worker-router.lock"
)


def _worker_runs_root(node: str) -> str:
    """Run root for a worker node. Legacy seat names resolve to their node."""
    node = kicker.NODE_FOR_LEGACY_SEAT.get(node, node)
    return {
        "delta": kicker.DELTA_RUNS,
        "charlie": kicker.CHARLIE_RUNS,
        "alpha": kicker.ALPHA_RUNS,
    }[node]


@contextmanager
def _worker_router_lock():
    os.makedirs(os.path.dirname(_WORKER_ROUTER_LOCK), mode=0o700, exist_ok=True)
    with open(_WORKER_ROUTER_LOCK, "a+", encoding="utf-8") as handle:
        os.chmod(_WORKER_ROUTER_LOCK, 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_worker_route(seat: str, token: str) -> dict:
    path = os.path.join(_worker_runs_root(seat), token, "worker-routing.json")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _worker_router_snapshot() -> dict:
    runs = kicker.status().get("runs", [])
    busy = {seat: [] for seat in _CLOUD_WORKER_SEATS}
    last_activity = {seat: 0.0 for seat in _CLOUD_WORKER_SEATS}
    active_routes = []
    for run in runs:
        seat = run.get("source")
        if seat not in busy:
            continue
        token = str(run.get("token") or "")
        try:
            modified = os.path.getmtime(os.path.join(_worker_runs_root(seat), token))
        except OSError:
            modified = 0.0
        last_activity[seat] = max(last_activity[seat], modified)
        if run.get("status") in _TERMINAL_RUN_STATES:
            continue
        compact = {
            "seat": seat,
            "token": token,
            "status": run.get("status"),
        }
        route = _read_worker_route(seat, token)
        if route:
            compact.update(
                target_host=route.get("target_host"),
                target_scope=route.get("target_scope"),
                claims=_route_claims(route),
                lane=route.get("lane"),
            )
            active_routes.append(compact.copy())
        busy[seat].append(compact)
    return {
        "busy": busy,
        "last_activity": last_activity,
        "active_routes": active_routes,
    }


def _parse_additional_scopes(raw: str) -> tuple:
    """Parse comma-separated `<host>:<scope>` claims into (claims, error).

    A build that mutates more than one host declares every host it touches
    rather than naming only the most consequential one. Each entry is
    `<target-host>:<repo|service|host|path>:<key>`; the first colon separates
    the host because no scope prefix is ever a host alias.
    """
    claims = []
    for entry in (part.strip() for part in (raw or "").split(",")):
        if not entry:
            continue
        host, _, scope = entry.partition(":")
        host = host.strip().lower()
        scope = scope.strip()
        if host not in _TARGET_HOSTS:
            return [], f"bad additional scope host: {host or entry}"
        if not _TARGET_SCOPE_RE.fullmatch(scope):
            return [], f"bad additional scope: {entry}"
        claims.append({"host": host, "scope": scope})
    return claims, ""


def _merge_claims(primary_host: str, primary_scope: str, extra: list) -> list:
    """The full claim set for a run, primary declaration first, deduplicated."""
    claims = []
    if primary_host:
        claims.append({"host": primary_host, "scope": primary_scope})
    claims.extend(extra or [])
    seen = set()
    merged = []
    for claim in claims:
        key = (claim.get("host"), claim.get("scope"))
        if key in seen:
            continue
        seen.add(key)
        merged.append({"host": claim.get("host"), "scope": claim.get("scope")})
    return merged


def _route_claims(route: dict) -> list:
    """Every (host, scope) pair a route holds, tolerating pre-claims routes."""
    claims = route.get("claims")
    if isinstance(claims, list) and claims:
        return [
            {"host": claim.get("host"), "scope": claim.get("scope")}
            for claim in claims
            if isinstance(claim, dict)
        ]
    return [{"host": route.get("target_host"), "scope": route.get("target_scope")}]


def _claims_collide(wanted: dict, held: dict) -> bool:
    """Two claims collide only on the same host; host:<alias> is host-global."""
    host = wanted.get("host")
    if not host or host != held.get("host"):
        return False
    host_scope = f"host:{host}"
    return (
        wanted.get("scope") == host_scope
        or held.get("scope") == host_scope
        or wanted.get("scope") == held.get("scope")
    )


def _scope_conflicts(
    target_host: str,
    target_scope: str,
    lane: str,
    active_routes: list,
    claims: list = None,
) -> list:
    if lane == "recon":
        return []
    wanted = list(claims or []) or [{"host": target_host, "scope": target_scope}]
    conflicts = []
    for route in active_routes:
        if route.get("lane") == "recon":
            continue
        held = _route_claims(route)
        if any(_claims_collide(w, h) for w in wanted for h in held):
            conflicts.append(route)
    return conflicts


def _select_worker(
    target_host: str, target_scope: str, lane: str, claims: list = None
) -> dict:
    snapshot = _worker_router_snapshot()
    conflicts = _scope_conflicts(
        target_host, target_scope, lane, snapshot["active_routes"], claims
    )
    if conflicts:
        return {
            "ok": False,
            "error": "target scope busy",
            "target_host": target_host,
            "target_scope": target_scope,
            "claims": claims or [{"host": target_host, "scope": target_scope}],
            "conflicts": conflicts,
        }
    free = [seat for seat in _CLOUD_WORKER_SEATS if not snapshot["busy"][seat]]
    if not free:
        return {
            "ok": False,
            "error": "no available worker",
            "busy": snapshot["busy"],
        }
    local_seat = next(
        (
            seat
            for seat, host in _WORKER_HOME_HOST.items()
            if host == target_host and seat in free
        ),
        None,
    )
    if local_seat:
        selected = local_seat
        reason = "target_local_worker_available"
    else:
        selected = min(
            free, key=lambda seat: (snapshot["last_activity"][seat], seat)
        )
        reason = "least_recently_used_available_worker"
    return {
        "ok": True,
        "seat": selected,
        "worker_host": _WORKER_HOME_HOST[selected],
        "target_host": target_host,
        "target_scope": target_scope,
        "claims": claims or [{"host": target_host, "scope": target_scope}],
        "cross_host": _WORKER_HOME_HOST[selected] != target_host,
        "reason": reason,
        "busy": snapshot["busy"],
    }


@mcp.tool()
def worker_routing_status(
    target_host: str = "",
    target_scope: str = "",
    lane: str = "recon",
    additional_scopes: str = "",
) -> dict:
    """Preview availability-aware worker routing without launching work.

    A target-local idle worker wins. If it is busy, the least-recently-used
    idle cloud worker wins and reaches the target through the verified SSH
    mesh. BUILD callers should provide a repo:/service:/path:/host: scope.
    `additional_scopes` previews a multi-host build as comma-separated
    `<host>:<scope>` claims; every claim is checked for collisions.
    """
    target_host = kicker.canonical_node((target_host or "").strip().lower())
    target_scope = (target_scope or "").strip()
    if target_host and target_host not in _TARGET_HOSTS:
        return {"ok": False, "error": "bad target host"}
    if lane not in kicker.LANES:
        return {"ok": False, "error": "bad lane"}
    if target_scope and not _TARGET_SCOPE_RE.fullmatch(target_scope):
        return {"ok": False, "error": "bad target scope"}
    extra_claims, scope_error = _parse_additional_scopes(additional_scopes)
    if scope_error:
        return {"ok": False, "error": scope_error}
    claims = _merge_claims(target_host, target_scope, extra_claims)
    if not target_host:
        return {"ok": True, **_worker_router_snapshot()}
    if lane == "prompts" and not target_scope:
        return {"ok": False, "error": "target scope required for build"}
    return _select_worker(target_host, target_scope, lane, claims)


@mcp.tool()
def dispatch(seat: str, lane: str, token: str, content: str, session_note: str = "", max_runtime_seconds: int = 0, provider: str = "auto", task_size: str = "small", target_host: str = "", target_scope: str = "", additional_scopes: str = "", localworker_eligible: bool = False) -> dict:
    """Atomically claim a relay token, preserve its immutable prompt, and launch it.

    This is the race-free replacement for separate stage_prompt + ask/summon
    calls. Legacy tools remain available for compatibility.

    Args:
        seat: auto, worker1, worker3, localworker, or worker2; worker4 is legacy compatibility only. Auto chooses
            Localworker first for an admitted compact Charlie-local deterministic
            task; otherwise it chooses an available cloud worker independently
            of the target host.
        lane: prompts for BUILD or recon for read-only investigation.
        token: A fresh strict FLEET relay token present verbatim in content.
        content: The complete prompt. It is stored once as request.md under the
            per-token run directory; shared latest.md is not read or changed.
        session_note: Optional session-note reference for the delivery footer.
        max_runtime_seconds: Optional wall-clock deadline from 1..604800;
            zero preserves the current unlimited behavior.
        provider: auto, claude, codex, or gemini. Cloud providers are available
            for Worker1, Worker3, and Worker2. Auto uses the central quota router.
        task_size: tiny, small, medium, or large. Used for quota prediction and
            the atomic reservation attached to this token.
        target_host: Physical host alias. Required with seat=auto.
        target_scope: Mutation collision key such as repo:/home/david/loupe,
            service:loupe.service, or host:delta. Required for BUILD with
            seat=auto. Different scopes on one host may run concurrently.
        additional_scopes: Every OTHER (host, scope) pair this run mutates, as
            comma-separated `<host>:<scope>` claims, e.g.
            "charlie:service:ollama.service,delta:repo:/home/david/loupe".
            A build that touches more than one host declares all of them here
            instead of under-declaring to the single most consequential host.
            Each claim is collision-checked and recorded in worker-routing.json.
        localworker_eligible: Opt into the fail-closed Localworker-first path.
            This is valid only with seat=auto, provider=auto, target_host=charlie,
            and a FLEET_COMPACT_DELIVERY_V1 envelope. The caller remains
            responsible for establishing that the task is deterministic,
            bounded, and fully specifiable before launch.
    """
    launchers = {
        "delta": kicker.kick_delta,
        "charlie": kicker.kick_charlie,
        "alpha": kicker.kick_alpha,
        # localworker is a provider on charlie, not a node; kept addressable while
        # provider routing is finished.
        "localworker": kicker.kick_localworker,
        "worker4": kicker.kick_worker4,
    }
    # WORKER1-1: seats are retired. Accept the legacy seat names as aliases so an
    # in-flight caller or an older instruction surface keeps working.
    seat = kicker.LAUNCHER_FOR_LEGACY_SEAT.get(seat, seat) if seat != "auto" else seat
    if seat != "auto" and seat not in launchers:
        return {"ok": False, "error": "bad seat"}
    if lane not in kicker.LANES:
        return {"ok": False, "error": "bad lane"}
    if not isinstance(content, str) or not content.strip():
        return {"ok": False, "error": "invalid prompt content"}
    if not isinstance(session_note, str):
        return {"ok": False, "error": "invalid session note"}
    if not isinstance(localworker_eligible, bool):
        return {"ok": False, "error": "bad localworker eligibility"}
    if provider not in kicker.PROVIDERS:
        return {"ok": False, "error": "bad provider"}
    if seat in ("localworker", "worker4") and provider in ("codex", "gemini"):
        return {"ok": False, "error": "provider unavailable for seat"}
    target_host = kicker.canonical_node((target_host or "").strip().lower())
    target_scope = (target_scope or "").strip()
    if target_host and target_host not in _TARGET_HOSTS:
        return {"ok": False, "error": "bad target host"}
    if target_scope and not _TARGET_SCOPE_RE.fullmatch(target_scope):
        return {"ok": False, "error": "bad target scope"}
    if seat == "auto" and not target_host:
        return {"ok": False, "error": "target host required for auto seat"}
    if seat == "auto" and lane == "prompts" and not target_scope:
        return {"ok": False, "error": "target scope required for auto build"}
    extra_claims, scope_error = _parse_additional_scopes(additional_scopes)
    if scope_error:
        return {"ok": False, "error": scope_error}
    claims = _merge_claims(target_host, target_scope, extra_claims)
    if localworker_eligible:
        if seat != "auto":
            return {"ok": False, "error": "localworker eligibility requires auto seat"}
        if provider != "auto":
            return {"ok": False, "error": "localworker eligibility requires auto provider"}
        if target_host != "charlie":
            return {"ok": False, "error": "localworker auto requires charlie target"}
        if (
            "FLEET_COMPACT_DELIVERY_V1_BEGIN" not in content
            or "FLEET_COMPACT_DELIVERY_V1_END" not in content
        ):
            return {
                "ok": False,
                "error": "localworker auto requires compact delivery contract",
            }
    kwargs = dict(
        lane=lane, session_note=session_note, prompt_content=content,
        max_runtime_seconds=max_runtime_seconds,
    )
    with _worker_router_lock():
        worker_route = None
        selected_seat = seat
        if seat == "auto" and localworker_eligible:
            selected_seat = "localworker"
            worker_route = {
                "ok": True,
                "seat": "localworker",
                "worker_host": "charlie",
                "target_host": "charlie",
                "target_scope": target_scope,
                "claims": claims,
                "cross_host": False,
                "reason": "localworker_first_compact_contract",
            }
        elif seat == "auto":
            worker_route = _select_worker(target_host, target_scope, lane, claims)
            if not worker_route.get("ok"):
                return worker_route
            selected_seat = worker_route["seat"]
        elif selected_seat in _CLOUD_WORKER_SEATS and target_host:
            snapshot = _worker_router_snapshot()
            conflicts = _scope_conflicts(
                target_host, target_scope, lane, snapshot["active_routes"], claims
            )
            if conflicts:
                return {
                    "ok": False,
                    "error": "target scope busy",
                    "target_host": target_host,
                    "target_scope": target_scope,
                    "claims": claims,
                    "conflicts": conflicts,
                }
            worker_route = {
                "ok": True,
                "seat": selected_seat,
                "worker_host": _WORKER_HOME_HOST.get(
                    kicker.NODE_FOR_LEGACY_SEAT.get(selected_seat, selected_seat),
                    selected_seat),
                "target_host": target_host,
                "target_scope": target_scope,
                "claims": claims,
                "cross_host": _WORKER_HOME_HOST.get(
                    kicker.NODE_FOR_LEGACY_SEAT.get(selected_seat, selected_seat),
                    selected_seat) != target_host,
                "reason": "explicit_worker",
            }
        if selected_seat in _CLOUD_WORKER_SEATS:
            kwargs["provider"] = provider
            kwargs["task_size"] = task_size
        if worker_route:
            worker_route.update(
                token=token,
                lane=lane,
                selected_at=datetime.now(timezone.utc).isoformat(),
            )
            transport = (
                "Use the local shell; never self-SSH."
                if not worker_route["cross_host"]
                else f"Use the configured `ssh {target_host}` physical-host alias."
            )
            kwargs["execution_context"] = "\n".join(
                (
                    f"- Selected worker: `{selected_seat}` on `{worker_route['worker_host']}`.",
                    f"- Target host: `{target_host}`.",
                    "- Declared scope claims: "
                    + (
                        ", ".join(
                            f"`{claim['host']}` \u2192 `{claim['scope']}`"
                            for claim in claims or worker_route.get("claims") or []
                            if claim.get("scope")
                        )
                        or "read-only/unlocked"
                    ) + ".",
                    f"- Transport: {transport}",
                    "- SSH is an execution transport only. It does not expand the "
                    "task's original authority, approval, paths, or stop conditions.",
                    "- Do not redirect the task to another host, broaden the scope, "
                    "or touch an overlapping repository/service.",
                )
            )
            kwargs["worker_routing"] = worker_route
        # a stored route or an older caller may still name a legacy seat
        selected_seat = kicker.LAUNCHER_FOR_LEGACY_SEAT.get(selected_seat, selected_seat)
        result = launchers[selected_seat](token, **kwargs)
        if worker_route and result.get("ok"):
            result["worker_routing"] = worker_route
        return result


@mcp.tool()
def delete_note(path: str) -> dict:
    """Soft-delete a .md note by moving it into ~/Vaults/.trash/ (never rm).

    Recoverable: the note is moved to .trash/<UTC-timestamp>/<original-path>.
    Refuses relay/ledger machinery (to-worker1, from-worker1, to-worker3, from-worker3,
    .trash, monthly journal shards, and response/latest lane files).

    Args:
        path: Vault-relative .md path under ~/Vaults to delete.

    Returns:
        {ok: true, original_path, trashed_to, deleted_at} on success,
        else {ok: false, error}.
    """
    return vs.delete_note(path)


_DISTILL_BACKEND_CMD = os.environ.get("DISTILL_BACKEND_CMD", "").strip()
_DISTILL_DEFAULT_SUMMARY_SENTENCES = 3
_DISTILL_DEFAULT_COMPRESS_LINES = 12
_DISTILL_MAX_STDOUT_BYTES = 64 * 1024
_DISTILL_MAX_STDERR_BYTES = 4 * 1024
distill_log = logging.getLogger("tower.distill")


def _distill_backend(operation: str, payload: dict) -> dict | None:
    """Call an explicitly configured Distill backend.

    The backend command is opt-in: an unset/empty DISTILL_BACKEND_CMD returns
    None so callers use Tower's deterministic fallback. The command receives
    the operation as argv[1] and JSON on stdin. Output is spooled to temporary
    files and only a bounded amount is decoded/returned.
    """
    if not _DISTILL_BACKEND_CMD:
        return None
    args = shlex.split(_DISTILL_BACKEND_CMD)
    if not args:
        return None
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            proc = subprocess.run(
                args + [operation],
                input=payload_bytes,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=45,
                check=False,
            )
            stdout_size = stdout_file.tell()
            stderr_size = stderr_file.tell()
            stdout_file.seek(0)
            stdout_raw = stdout_file.read(_DISTILL_MAX_STDOUT_BYTES + 1)
            stderr_file.seek(max(0, stderr_size - _DISTILL_MAX_STDERR_BYTES))
            stderr_raw = stderr_file.read(_DISTILL_MAX_STDERR_BYTES)
    except Exception as e:
        return {
            "ok": False,
            "backend": "command",
            "operation": operation,
            "error": "backend exception",
            "error_type": type(e).__name__,
        }
    if stdout_size > _DISTILL_MAX_STDOUT_BYTES:
        return {
            "ok": False,
            "backend": "command",
            "operation": operation,
            "error": "backend output exceeded limit",
            "stdout_bytes": stdout_size,
        }
    stderr = stderr_raw.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return {
            "ok": False,
            "backend": "command",
            "operation": operation,
            "error": "backend failed",
            "returncode": proc.returncode,
            "stderr": stderr[-1200:],
        }
    stdout = stdout_raw.decode("utf-8", errors="replace").strip()
    if not stdout:
        return {
            "ok": False,
            "backend": "command",
            "operation": operation,
            "error": "backend returned empty output",
        }
    try:
        data = json.loads(stdout)
    except Exception:
        return {
            "ok": False,
            "backend": "command",
            "operation": operation,
            "error": "backend output was not valid JSON",
            "stdout": stdout[-1200:],
        }
    if not isinstance(data, dict):
        data = {"result": data}
    data.setdefault("ok", True)
    data.setdefault("backend", "command")
    data.setdefault("operation", operation)
    return data

def _distill_sentences(text: str) -> list[str]:
    chunks = []
    for para in re.split(r"\n\s*\n+", text or ""):
        para = " ".join(line.strip() for line in para.splitlines() if line.strip())
        if not para:
            continue
        parts = re.split(r"(?<=[.!?])\s+", para)
        for part in parts:
            part = part.strip()
            if part:
                chunks.append(part)
    return chunks


def _distill_extract_key_lines(text: str, limit: int) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "•", ">", "#")):
            lines.append(line)
        elif ":" in line and len(line) <= 180:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _distill_fallback_summary(text: str, max_sentences: int = _DISTILL_DEFAULT_SUMMARY_SENTENCES) -> dict:
    text = (text or "").strip()
    sentences = _distill_sentences(text)
    key_lines = _distill_extract_key_lines(text, limit=4)
    summary_parts = []
    if key_lines:
        summary_parts.extend(key_lines[:2])
    for sentence in sentences:
        if len(summary_parts) >= max_sentences:
            break
        if sentence not in summary_parts:
            summary_parts.append(sentence)
    summary = " ".join(summary_parts[:max_sentences]).strip()
    if not summary and text:
        summary = textwrap.shorten(" ".join(text.split()), width=400, placeholder="…")
    return {
        "ok": True,
        "backend": "fallback",
        "operation": "summarize",
        "strategy": "extractive",
        "summary": summary,
        "summary_sentences": min(len(summary_parts), max_sentences),
        "input_chars": len(text),
    }


def _distill_fallback_compress(text: str, max_lines: int = _DISTILL_DEFAULT_COMPRESS_LINES) -> dict:
    text = (text or "").strip()
    lines = []
    seen = set()
    for raw in text.splitlines():
        line = " ".join(raw.split()).strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [f"[truncated {len(lines) - (max_lines - 1)} extra line(s)]"]
    compressed = "\n".join(lines)
    if not compressed and text:
        compressed = textwrap.shorten(" ".join(text.split()), width=500, placeholder="…")
    return {
        "ok": True,
        "backend": "fallback",
        "operation": "compress",
        "strategy": "dedupe-lines",
        "compressed": compressed,
        "input_chars": len(text),
        "output_chars": len(compressed),
    }


def _distill_route_fallback(text: str, hint: str = "") -> dict:
    blob = f"{hint}\n{text}".lower()
    rules = [
        ("fleet/ops", ["ssh", "service", "daemon", "systemd", "reboot", "netplan", "nat", "gpu", "mount", "lvm"]),
        ("vault/search", ["vault", "note", "md", "obsidian", "search", "decision", "session", "log"]),
        ("code/change", ["patch", "diff", "commit", "bug", "fix", "function", "class", "tool"]),
        ("summarize", ["long", "summary", "digest", "brief", "reduce", "condense", "compress"]),
        ("extract", ["extract", "parse", "fields", "json", "schema", "table"]),
    ]
    score = []
    for label, words in rules:
        hits = sum(1 for w in words if w in blob)
        if hits:
            score.append((hits, label))
    score.sort(reverse=True)
    decision = score[0][1] if score else "unknown"
    handoff = {
        "fleet/ops": "use a focused ops seat or direct shell verification",
        "vault/search": "use search_vault/get_note or a search-backed reduction",
        "code/change": "route to a code patch seat with a tight diff",
        "summarize": "call distill_summarize",
        "extract": "call distill_extract",
        "unknown": "send to a larger model with a fresh prompt",
    }[decision]
    return {
        "ok": True,
        "backend": "fallback",
        "operation": "route",
        "strategy": "keyword",
        "decision": decision,
        "handoff": handoff,
        "confidence": 0.55 if score else 0.2,
        "matches": [{"target": label, "hits": hits} for hits, label in score[:3]],
    }


def _distill_fallback_extract(text: str, fields: list[str] | None = None) -> dict:
    key_lines = _distill_extract_key_lines(text, limit=10)
    return {
        "ok": True,
        "backend": "fallback",
        "operation": "extract",
        "hostnames": [line for line in key_lines if "." in line or "/" not in line][:5] if not fields or "hostnames" in fields else [],
        "paths": [line for line in key_lines if "/" in line][:5] if not fields or "paths" in fields else [],
        "errors": [line for line in key_lines if re.search(r"\b(error|failed|traceback|exception|oom|denied)\b", line, re.I)][:5] if not fields or "errors" in fields else [],
        "timestamps": [line for line in key_lines if re.search(r"\b\d{4}-\d{2}-\d{2}\b", line)][:5] if not fields or "timestamps" in fields else [],
        "ids": [line for line in key_lines if "FLEET-" in line][:5] if not fields or "ids" in fields else [],
        "commands": [line for line in key_lines if line.startswith(("ssh ", "sudo ", "curl ", "ollama ", "$"))][:5] if not fields or "commands" in fields else [],
        "input_chars": len((text or "")),
    }


def _distill_resolve_backend(operation: str, backend: dict | None, fallback_factory) -> dict:
    """Return a successful backend result or a deterministic fallback.

    Logs only the branch and bounded error class/code; never logs payload text.
    """
    if backend is None:
        distill_log.info("distill operation=%s branch=fallback reason=not_configured", operation)
        result = fallback_factory()
        result["fallback_reason"] = "backend_not_configured"
        return result
    if backend.get("ok", True):
        distill_log.info("distill operation=%s branch=backend", operation)
        return backend
    distill_log.warning(
        "distill operation=%s branch=fallback reason=backend_error error=%s",
        operation,
        backend.get("error", "unknown"),
    )
    result = fallback_factory()
    result["fallback_reason"] = "backend_error"
    result["backend_error"] = {
        key: backend[key]
        for key in ("error", "error_type", "returncode", "stdout_bytes")
        if key in backend
    }
    return result


@mcp.tool()
def distill_summarize(text: str, max_bullets: int = 5, focus: str = "general") -> dict:
    """Summarize a blob for downstream use.

    DISTILL_BACKEND_CMD is opt-in. When it is absent or its command fails, Tower
    returns a deterministic extractive fallback with a non-payload reason.
    """
    backend = _distill_backend("summarize", {"text": text, "max_bullets": max_bullets, "focus": focus})
    return _distill_resolve_backend(
        "summarize",
        backend,
        lambda: _distill_fallback_summary(text, max_sentences=max(1, int(max_bullets or 5))),
    )


@mcp.tool()
def distill_compress(text: str, audience: str = "Claude", goal: str = "") -> dict:
    """Compress text; use the opt-in backend or deterministic fallback."""
    backend = _distill_backend("compress", {"text": text, "audience": audience, "goal": goal})
    return _distill_resolve_backend("compress", backend, lambda: _distill_fallback_compress(text))


@mcp.tool()
def distill_route(text: str, candidates: list[str] | None = None, prefer_local: bool = True) -> dict:
    """Route a request; use the opt-in backend or deterministic fallback."""
    backend = _distill_backend("route", {"text": text, "candidates": candidates, "prefer_local": prefer_local})
    hint = ", ".join(candidates or [])
    return _distill_resolve_backend("route", backend, lambda: _distill_route_fallback(text, hint=hint))


@mcp.tool()
def distill_extract(text: str, fields: list[str] | None = None) -> dict:
    """Extract structure; use the opt-in backend or deterministic fallback."""
    backend = _distill_backend("extract", {"text": text, "fields": fields})
    return _distill_resolve_backend("extract", backend, lambda: _distill_fallback_extract(text, fields))


@mcp.tool()
def status(token: str = "") -> dict:
    """Poll relay runs. Never blocks; reads only the filesystem.

    With a token: that run's state (running/died/done), whichever seat ran it.
    Without a token: all in-flight/recent runs across Worker1 and Worker3.

    Args:
        token: The relay token passed to summon_worker1/ask_worker1. Omit to list all runs.

    Returns:
        With a token: {ok:false, status:"missing"} if unknown; {ok:true,
        status:"running", pid, started_at} while in flight; {ok:true,
        status:"done", exit_code, transcript, transcript_bytes,
        response_token_match} once finished.
        Without a token: {ok:true, runs:[{token, status, started_at, pid}, ...]}.
    """
    return kicker.status(token or None)


@mcp.tool()
def inspect_run(
    token: str,
    include: list[str] | None = None,
    max_events: int = 20,
    max_bytes: int = 65536,
) -> dict:
    """Inspect one relay run with strict transcript and response bounds.

    Args:
        token: Exact strict FLEET run token.
        include: Any of status, response, transcript_tail, tool_results.
        max_events: Maximum parsed tail events/tool results (clamped 1..100).
        max_bytes: Total returned source-text budget and transcript tail scan
            cap (clamped 1024..262144 bytes).

    Returns status plus requested durable response/token proof, compact tail
    events, selected tool-result evidence, and explicit truncation metadata.
    It never mutates run state and never loads a whole large transcript.
    """
    return kicker.inspect_run(token, include, max_events, max_bytes)


@mcp.tool()
def relay_audit(max_recent: int = 20) -> dict:
    """Read-only relay health audit across active seats plus retired Worker4 compatibility state.

    Unlike status(), which answers "what happened to run X", this answers "is
    the relay itself healthy": for each seat it reports whether the run root
    and its from-<seat> relay root exist, the total run count, counts by status
    (done/running/launching/stale), the stale count, and the single most recent
    run's metadata (and most recent completed-run timestamp, if any). It also
    reports the Syncthing service state via a bounded, read-only `systemctl
    is-active` check — Syncthing is only the transport that carries run-state
    files back to this host, NOT proof that a remote worker process is alive;
    a seat can show "active" Syncthing and still have every run gone stale.

    Never mutates runs, prompts, notes, or services, and never inspects
    transcripts/prompts/stderr. `max_recent` (clamped to 1..100) bounds how many
    of each seat's most-recently-modified run dirs are opened/classified per
    call. `inspected_run_count`, `inspection_limit`, `status_counts_scope`, and
    root `scope` make explicit that status/stale counts cover that recent slice,
    while `total_run_count` counts every run directory.

    Args:
        max_recent: Per-seat cap on recently-modified run dirs inspected (1..100, default 20).

    Returns:
        {ok, generated_at, max_recent, sync_dependency, syncthing_service,
        vault_root, vault_root_exists, relay_root, relay_root_exists,
        seats: [{seat, run_root, run_root_exists, from_seat_root,
        from_seat_root_exists, total_run_count, inspected_run_count,
        inspection_limit, status_counts_scope, status_counts, stale_count,
        most_recent_run, most_recent_done_at}, ...]}.
    """
    return kicker.relay_audit(max_recent)


def _listen_sockets(addresses, port):
    """One listening socket per configured address.

    A per-address failure is a warning, not a fatal: a tailnet address that is
    not up yet (tailscaled still starting after a reboot) must never be able to
    take the whole Tower down, since loopback carries the tunnel origin. Fails
    hard only if NOTHING could be bound. Same tolerance as breadcrumbs' listener
    loop."""
    socks = []
    for addr in addresses:
        try:
            info = socket.getaddrinfo(
                addr, port, proto=socket.IPPROTO_TCP, flags=socket.AI_PASSIVE
            )[0]
            s = socket.socket(info[0], info[1])
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if info[0] == socket.AF_INET6:
                # keep v6 sockets v6-only so a wildcard v6 bind can't silently
                # also claim v4 and widen reachability past the explicit list
                s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            s.bind(info[4])
            s.listen()
            s.set_inheritable(True)
            socks.append(s)
            log.info("tower: listening on %s:%d", addr, port)
        except OSError as e:
            log.warning(
                "tower: could not bind %s:%d (%s) — continuing without it",
                addr, port, type(e).__name__,
            )
    return socks


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = mcp.streamable_http_app()
    app.add_middleware(CfAccessJWTMiddleware)
    log.info(
        "tower starting with Cloudflare Access JWT gate enabled "
        "(trusted channels: loopback, tailnet; bind: %s)",
        ",".join(BIND_ADDRESSES),
    )
    _socks = _listen_sockets(BIND_ADDRESSES, TOWER_PORT)
    if not _socks:
        raise SystemExit(
            f"tower: no listening socket could be bound from {BIND_ADDRESSES}"
        )
    _config = uvicorn.Config(app, log_level="info")
    uvicorn.Server(_config).run(sockets=_socks)

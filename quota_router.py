"""Quota-aware provider selection shared by Tower and Panel.

The router reads only sanitized quota snapshots, keeps account-wide reservations
in one Alpha-local SQLite ledger, and returns an auditable score breakdown.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROVIDERS = ("claude", "codex", "gemini")
# Codex exposes only a weekly quota window; it has no five-hour telemetry to
# be "unknown" or "missing" -- the dimension is structurally not applicable.
FIVE_HOUR_CAPABLE = {"claude": True, "codex": False, "gemini": True}
LANES = ("worker",)
SIZES = ("tiny", "small", "medium", "large")
STATE_RANK = {"RED": 0, "AMBER": 1, "YELLOW": 2, "GREEN": 3}
MODELS = {
    "worker": {
        "claude": "sonnet-5",
        "codex": "gpt-5.6-terra",
        "gemini": "gemini-3.6-flash-high",
        "localworker": "gpt-oss:20b",
    },
}
DEFAULT_COSTS = {
    "tiny": (0.5, 3.0),
    "small": (1.5, 8.0),
    "medium": (3.0, 18.0),
    "large": (6.0, 32.0),
}
STALE_SECONDS = 900
_COUNTDOWN_RE = re.compile(
    r"^\s*(?:(\d+)d\s*)?(?:(\d+)h\s*)?(?:(\d+)m\s*)?$",
    re.IGNORECASE,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _iso_epoch(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _countdown_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    match = _COUNTDOWN_RE.fullmatch(value)
    if not match or not any(match.groups()):
        return None
    days, hours, minutes = (int(part or 0) for part in match.groups())
    return float(days * 86400 + hours * 3600 + minutes * 60)


def _used(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return _clamp(float(value), 0.0, 100.0)
    return None


class QuotaRouter:
    def __init__(
        self,
        quota_root: str | os.PathLike[str] | None = None,
        db_path: str | os.PathLike[str] | None = None,
        gemini_worker_marker: str | os.PathLike[str] | None = None,
        now_fn=time.time,
    ) -> None:
        home = Path.home()
        self.quota_root = Path(
            quota_root
            or home / "Vaults" / "homelab-vault" / "heartbeats" / "quota"
        )
        self.db_path = Path(
            db_path
            or home / ".local" / "state" / "fleet" / "quota-router.sqlite3"
        )
        self.gemini_worker_marker = Path(
            gemini_worker_marker
            or home
            / ".config"
            / "fleet"
            / "gemini-headless-worker-enabled"
        )
        self.now_fn = now_fn
        self._initialize_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize_db(self) -> None:
        connection = self._connect()
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                lane TEXT NOT NULL,
                size TEXT NOT NULL,
                q_week REAL NOT NULL,
                q_five REAL NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                released_at REAL,
                decision_json TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(reservations)").fetchall()
        }
        for name, declaration in (
            ("release_reason", "TEXT"),
            ("actual_week", "REAL"),
            ("actual_five", "REAL"),
        ):
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE reservations ADD COLUMN {name} {declaration}"
                )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_health (
                provider TEXT PRIMARY KEY,
                failure_streak INTEGER NOT NULL DEFAULT 0,
                cooldown_until REAL,
                last_reason TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reservation_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                size TEXT NOT NULL,
                q_week REAL,
                q_five REAL,
                success INTEGER NOT NULL,
                observed_at REAL NOT NULL,
                UNIQUE(reservation_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS reservations_active "
            "ON reservations(provider, released_at, expires_at)"
        )
        connection.commit()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass
        connection.close()

    def _model_usage(self, now: float) -> tuple[dict[str, Any] | None, str | None]:
        path = self.quota_root / "model-usage.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            generated = _iso_epoch(payload.get("generated_at"))
            if generated is None or now - generated > STALE_SECONDS:
                return None, "telemetry_stale"
            if not payload.get("ok"):
                return None, "telemetry_unavailable"
            return payload, None
        except (OSError, ValueError, TypeError):
            return None, "telemetry_missing"

    def _codex_snapshot(self, now: float) -> tuple[dict[str, Any] | None, str | None]:
        freshest: tuple[float, dict[str, Any]] | None = None
        for path in self.quota_root.glob("*-codex.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not payload.get("ok"):
                    continue
                generated = _iso_epoch(payload.get("generated_at"))
                if generated is None or now - generated > STALE_SECONDS:
                    continue
                if freshest is None or generated > freshest[0]:
                    freshest = (generated, payload)
            except (OSError, ValueError, TypeError):
                continue
        if freshest is None:
            return None, "telemetry_unavailable"
        return freshest[1], None

    @staticmethod
    def _window_from_usage(
        window: dict[str, Any] | None,
        generated: float,
        approximate: bool = False,
    ) -> dict[str, Any]:
        window = window or {}
        used = _used(window.get("used_percent"))
        reset = _iso_epoch(window.get("resets_at"))
        if reset is None:
            seconds = _countdown_seconds(
                window.get("refreshes_in") or window.get("resets")
            )
            if seconds is not None:
                reset = generated + seconds
                approximate = True
        return {
            "used": used,
            "remaining": None if used is None else 100.0 - used,
            "reset_at": reset,
            "approximate": approximate,
        }

    @staticmethod
    def _window_from_codex(window: dict[str, Any] | None) -> dict[str, Any]:
        window = window or {}
        used = _used(window.get("usedPercent"))
        reset = window.get("resetsAt")
        return {
            "used": used,
            "remaining": None if used is None else 100.0 - used,
            "reset_at": float(reset) if isinstance(reset, (int, float)) else None,
            "approximate": False,
        }

    def _telemetry(self, now: float) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        model_usage, model_error = self._model_usage(now)
        if model_usage is None:
            for provider in ("claude", "gemini"):
                result[provider] = {"ok": False, "error": model_error}
        else:
            generated = _iso_epoch(model_usage.get("generated_at")) or now
            claude = model_usage.get("claude") or {}
            claude_windows = claude.get("windows") or {}
            result["claude"] = {
                "ok": bool(claude.get("ok")),
                "source": claude.get("source", "model-usage"),
                "weekly": self._window_from_usage(
                    claude_windows.get("weekly"), generated
                ),
                "five_hour": self._window_from_usage(
                    claude_windows.get("five_hour"), generated
                ),
                "five_hour_applicable": True,
                "confidence": 1.0,
                "generated_at": generated,
            }
            gemini = model_usage.get("gemini") or {}
            gemini_windows = gemini.get("windows") or {}
            result["gemini"] = {
                "ok": bool(gemini.get("ok")),
                "source": gemini.get("source", "agy /usage"),
                "weekly": self._window_from_usage(
                    gemini_windows.get("weekly"), generated, approximate=True
                ),
                "five_hour": self._window_from_usage(
                    gemini_windows.get("five_hour"), generated, approximate=True
                ),
                "five_hour_applicable": True,
                "confidence": 0.85,
                "generated_at": generated,
            }

        codex, codex_error = self._codex_snapshot(now)
        if codex is None:
            result["codex"] = {"ok": False, "error": codex_error}
        else:
            limits = codex.get("rateLimits") or {}
            windows = [
                window
                for window in (limits.get("primary"), limits.get("secondary"))
                if isinstance(window, dict)
            ]
            weekly = next(
                (
                    window
                    for window in windows
                    if window.get("windowDurationMins") == 10080
                ),
                None,
            )
            five = next(
                (
                    window
                    for window in windows
                    if window.get("windowDurationMins") == 300
                ),
                None,
            )
            result["codex"] = {
                "ok": not bool(
                    limits.get("rateLimitReachedType")
                    or limits.get("spendControlReached")
                ),
                "source": "codex app-server",
                "weekly": self._window_from_codex(weekly),
                "five_hour": self._window_from_codex(five),
                "five_hour_applicable": False,
                # Codex has no five-hour window at all, so its absence is not
                # a data-quality signal -- never penalize confidence for it.
                "confidence": 1.0,
                "generated_at": _iso_epoch(codex.get("generated_at")) or now,
            }
        return result

    @staticmethod
    def _hours_until(reset_at: float | None, now: float, cap: float) -> float:
        if reset_at is None:
            return cap
        return _clamp((reset_at - now) / 3600.0, 0.0, cap)

    @staticmethod
    def _reserves(lane: str, t_week: float, t_five: float) -> tuple[float, float]:
        return 5.0 + 15.0 * (t_week / 168.0), 5.0 + 10.0 * (t_five / 5.0)

    def _active_reservations(
        self, connection: sqlite3.Connection, now: float
    ) -> dict[str, tuple[float, float]]:
        connection.execute(
            "UPDATE reservations SET released_at = ?, release_reason = 'expired' "
            "WHERE released_at IS NULL AND expires_at <= ?",
            (now, now),
        )
        rows = connection.execute(
            "SELECT provider, COALESCE(SUM(q_week), 0) AS q_week, "
            "COALESCE(SUM(q_five), 0) AS q_five "
            "FROM reservations WHERE released_at IS NULL AND expires_at > ? "
            "GROUP BY provider",
            (now,),
        ).fetchall()
        return {
            row["provider"]: (float(row["q_week"]), float(row["q_five"]))
            for row in rows
        }

    def _predicted_cost(
        self, connection: sqlite3.Connection, provider: str, size: str
    ) -> tuple[float, float]:
        default_week, default_five = DEFAULT_COSTS[size]
        five_hour_capable = FIVE_HOUR_CAPABLE.get(provider, True)
        if not five_hour_capable:
            # No five-hour window exists to spend against; never apply the
            # generic five-hour default cost to a weekly-only provider.
            default_five = 0.0
        rows = connection.execute(
            "SELECT q_week, q_five FROM cost_observations "
            "WHERE provider = ? AND size = ? AND success = 1 "
            "ORDER BY observed_at DESC LIMIT 20",
            (provider, size),
        ).fetchall()
        if len(rows) < 3:
            return default_week, default_five

        def calibrated(name: str, default: float) -> float:
            values = [float(row[name]) for row in rows if row[name] is not None]
            if len(values) < 3:
                return default
            average = sum(values) / len(values)
            return max(default * 0.25, min(default * 4.0, average))

        calibrated_week = calibrated("q_week", default_week)
        calibrated_five = (
            0.0 if not five_hour_capable else calibrated("q_five", default_five)
        )
        return calibrated_week, calibrated_five

    def _cooldown(
        self, connection: sqlite3.Connection, provider: str, now: float
    ) -> tuple[float | None, str | None]:
        row = connection.execute(
            "SELECT cooldown_until, last_reason FROM provider_health WHERE provider = ?",
            (provider,),
        ).fetchone()
        if row is None or row["cooldown_until"] is None:
            return None, None
        until = float(row["cooldown_until"])
        return (until, row["last_reason"]) if until > now else (None, None)

    def _candidate(
        self,
        provider: str,
        lane: str,
        size: str,
        telemetry: dict[str, Any],
        reserved: tuple[float, float],
        predicted_cost: tuple[float, float],
        cooldown: tuple[float | None, str | None],
        now: float,
    ) -> dict[str, Any]:
        q_week, q_five = predicted_cost
        five_hour_applicable = FIVE_HOUR_CAPABLE.get(provider, True)
        base = {
            "provider": provider,
            "model": MODELS[lane][provider],
            "lane": lane,
            "size": size,
            "predicted_cost": {"weekly": q_week, "five_hour": q_five},
            "reserved": {"weekly": reserved[0], "five_hour": reserved[1]},
            "five_hour_applicable": five_hour_applicable,
        }
        if cooldown[0] is not None:
            return {
                **base,
                "state": "RED",
                "score": 0.0,
                "reason": "provider_cooldown",
                "cooldown_until": cooldown[0],
                "cooldown_reason": cooldown[1],
            }
        if not telemetry.get("ok"):
            return {
                **base,
                "state": "RED",
                "score": 0.0,
                "reason": telemetry.get("error", "provider_unavailable"),
            }
        if (
            provider == "gemini"
            and lane == "worker"
            and not self.gemini_worker_marker.is_file()
        ):
            return {
                **base,
                "state": "RED",
                "score": 0.0,
                "reason": "headless_permissions_unconfigured",
            }
        weekly = telemetry.get("weekly") or {}
        five = telemetry.get("five_hour") or {}
        weekly_remaining = weekly.get("remaining")
        five_remaining = five.get("remaining")
        if weekly_remaining is None:
            return {
                **base,
                "state": "RED",
                "score": 0.0,
                "reason": "weekly_window_missing",
            }
        t_week = self._hours_until(weekly.get("reset_at"), now, 168.0)
        t_five = self._hours_until(five.get("reset_at"), now, 5.0)
        reserve_week, reserve_five = self._reserves(lane, t_week, t_five)
        post_week = weekly_remaining - reserved[0] - q_week
        post_five = (
            None
            if five_remaining is None
            else five_remaining - reserved[1] - q_five
        )
        if post_week < 0 or (post_five is not None and post_five < 0):
            state = "RED"
            reason = "predicted_capacity_insufficient"
        elif post_five is None:
            if five_hour_applicable:
                state = "YELLOW" if post_week >= reserve_week else "AMBER"
                reason = (
                    "five_hour_unknown" if state == "YELLOW" else "weekly_reserve_breach"
                )
            else:
                # Weekly-only provider: no five-hour dimension to be unknown
                # about, so a healthy weekly reserve is simply GREEN.
                state = "GREEN" if post_week >= reserve_week else "AMBER"
                reason = (
                    "within_weekly_reserve"
                    if state == "GREEN"
                    else "weekly_reserve_breach"
                )
        elif post_week >= reserve_week and post_five >= reserve_five:
            state = "GREEN"
            reason = "within_dynamic_reserves"
        else:
            state = "AMBER"
            reason = "dynamic_reserve_breach"

        weekly_safety = _clamp(
            (post_week - reserve_week) / max(1.0, 100.0 - reserve_week)
        )
        confidence = float(telemetry.get("confidence", 0.5))
        if not five_hour_applicable:
            # Score entirely from the supported weekly telemetry -- no
            # synthetic five-hour safety value and no confidence penalty for
            # a dimension that structurally does not exist for this provider.
            expiry = (weekly_remaining / 100.0) * (1.0 - t_week / 168.0)
            score = (
                60.0 * weekly_safety
                + 15.0 * expiry
                + 15.0
                + 5.0 * confidence
                + 5.0
            )
            if state == "RED":
                score = 0.0
            return {
                **base,
                "state": state,
                "score": round(score, 2),
                "reason": reason,
                "source": telemetry.get("source"),
                "confidence": confidence,
                "remaining": {
                    "weekly": round(weekly_remaining, 2),
                    "five_hour": None,
                },
                "post_task": {
                    "weekly": round(post_week, 2),
                    "five_hour": None,
                },
                "reserve": {
                    "weekly": round(reserve_week, 2),
                    "five_hour": round(reserve_five, 2),
                },
                "resets_at": {
                    "weekly": weekly.get("reset_at"),
                    "five_hour": five.get("reset_at"),
                },
            }
        if post_five is None:
            five_safety = 0.50
        else:
            five_safety = _clamp(
                (post_five - reserve_five) / max(1.0, 100.0 - reserve_five)
            )
        expiry = 0.0
        if five_remaining is not None:
            expiry += (
                0.65
                * (five_remaining / 100.0)
                * (1.0 - t_five / 5.0)
            )
        expiry += (
            0.35
            * (weekly_remaining / 100.0)
            * (1.0 - t_week / 168.0)
        )
        score = (
            35.0 * weekly_safety
            + 25.0 * five_safety
            + 15.0 * expiry
            + 15.0
            + 5.0 * confidence
            + 5.0
        )
        if state == "RED":
            score = 0.0
        return {
            **base,
            "state": state,
            "score": round(score, 2),
            "reason": reason,
            "source": telemetry.get("source"),
            "confidence": confidence,
            "remaining": {
                "weekly": round(weekly_remaining, 2),
                "five_hour": (
                    None if five_remaining is None else round(five_remaining, 2)
                ),
            },
            "post_task": {
                "weekly": round(post_week, 2),
                "five_hour": None if post_five is None else round(post_five, 2),
            },
            "reserve": {
                "weekly": round(reserve_week, 2),
                "five_hour": round(reserve_five, 2),
            },
            "resets_at": {
                "weekly": weekly.get("reset_at"),
                "five_hour": five.get("reset_at"),
            },
        }

    def recommend(
        self,
        *,
        lane: str = "worker",
        size: str = "small",
        allowed_providers: Iterable[str] | None = None,
        explicit_provider: str | None = None,
        current_provider: str | None = None,
        localworker_eligible: bool = False,
        reserve: bool = False,
        reservation_id: str | None = None,
        ttl_seconds: int = 22500,
    ) -> dict[str, Any]:
        lane = (lane or "").strip().lower()
        size = (size or "").strip().lower()
        if lane not in LANES:
            return {"ok": False, "error": "bad lane"}
        if size not in SIZES:
            return {"ok": False, "error": "bad task size"}
        allowed = tuple(
            dict.fromkeys(
                provider.strip().lower()
                for provider in (allowed_providers or PROVIDERS)
            )
        )
        if not allowed or any(provider not in PROVIDERS for provider in allowed):
            return {"ok": False, "error": "bad allowed provider"}
        explicit = (explicit_provider or "").strip().lower() or None
        current = (current_provider or "").strip().lower() or None
        if explicit is not None and explicit not in allowed:
            return {"ok": False, "error": "explicit provider unavailable"}
        if reserve and not reservation_id:
            return {"ok": False, "error": "reservation id required"}
        if not isinstance(ttl_seconds, int) or not 60 <= ttl_seconds <= 604800:
            return {"ok": False, "error": "bad reservation ttl"}
        if localworker_eligible and explicit is None:
            return {
                "ok": True,
                "provider": "localworker",
                "model": MODELS[lane]["localworker"],
                "state": "GREEN",
                "score": 100.0,
                "reason": "zero_cloud_localworker_first",
                "reserved": False,
                "candidates": [],
            }

        now = float(self.now_fn())
        telemetry = self._telemetry(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = self._active_reservations(connection, now)
            candidates = [
                self._candidate(
                    provider,
                    lane,
                    size,
                    telemetry.get(provider, {"ok": False, "error": "missing"}),
                    active.get(provider, (0.0, 0.0)),
                    self._predicted_cost(connection, provider, size),
                    self._cooldown(connection, provider, now),
                    now,
                )
                for provider in allowed
            ]
            if explicit is not None:
                winner = next(item for item in candidates if item["provider"] == explicit)
                if winner["state"] == "RED":
                    connection.rollback()
                    return {
                        "ok": False,
                        "error": f"{explicit}:{winner['reason']}",
                        "candidates": candidates,
                    }
                winner = {**winner, "reason": f"explicit_pin:{winner['reason']}"}
            else:
                eligible = [
                    item for item in candidates if item["state"] != "RED"
                ]
                if not eligible:
                    connection.rollback()
                    return {
                        "ok": False,
                        "error": "no eligible cloud provider",
                        "candidates": candidates,
                    }
                best_rank = max(STATE_RANK[item["state"]] for item in eligible)
                top_state = [
                    item
                    for item in eligible
                    if STATE_RANK[item["state"]] == best_rank
                ]
                winner = max(top_state, key=lambda item: item["score"])
                if current in allowed:
                    current_item = next(
                        item for item in top_state if item["provider"] == current
                    ) if any(item["provider"] == current for item in top_state) else None
                    if (
                        current_item is not None
                        and winner["score"] - current_item["score"] < 8.0
                    ):
                        winner = {
                            **current_item,
                            "reason": f"hysteresis_keep_current:{current_item['reason']}",
                        }
                winner = {
                    **winner,
                    "reason": (
                        winner["reason"]
                        if winner["reason"].startswith("hysteresis_")
                        else f"auto_{winner['state'].lower()}:{winner['reason']}"
                    ),
                }

            decision = {
                "ok": True,
                "provider": winner["provider"],
                "model": winner["model"],
                "state": winner["state"],
                "score": winner["score"],
                "reason": winner["reason"],
                "generated_at": datetime.fromtimestamp(
                    now, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lane": lane,
                "size": size,
                "candidates": candidates,
                "reserved": False,
            }
            if reserve:
                q_week = float(winner["predicted_cost"]["weekly"])
                q_five = float(winner["predicted_cost"]["five_hour"])
                try:
                    connection.execute(
                        "INSERT INTO reservations "
                        "(reservation_id, provider, model, lane, size, q_week, q_five, "
                        "created_at, expires_at, released_at, decision_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                        (
                            reservation_id,
                            winner["provider"],
                            winner["model"],
                            lane,
                            size,
                            q_week,
                            q_five,
                            now,
                            now + ttl_seconds,
                            json.dumps(decision, sort_keys=True),
                        ),
                    )
                except sqlite3.IntegrityError:
                    connection.rollback()
                    return {"ok": False, "error": "reservation already exists"}
                decision["reserved"] = True
                decision["reservation_id"] = reservation_id
                decision["reservation_expires_at"] = now + ttl_seconds
            connection.commit()
            return decision
        finally:
            connection.close()

    def release(self, reservation_id: str, reason: str = "terminal_success") -> bool:
        if not reservation_id:
            return False
        now = float(self.now_fn())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ? "
                "AND released_at IS NULL",
                (reservation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            success = reason == "terminal_success"
            actual_week = actual_five = None
            try:
                decision = json.loads(row["decision_json"])
                winner = next(
                    item for item in decision.get("candidates", [])
                    if item.get("provider") == row["provider"]
                )
                current = self._telemetry(now).get(row["provider"], {})
                start = winner.get("remaining") or {}
                weekly = (current.get("weekly") or {}).get("remaining")
                five = (current.get("five_hour") or {}).get("remaining")
                if isinstance(start.get("weekly"), (int, float)) and isinstance(weekly, (int, float)):
                    delta = float(start["weekly"]) - float(weekly)
                    # Quota snapshots are rounded. A zero delta is not evidence
                    # that a run was free, so never train the forecast on it.
                    if 0 < delta <= 50:
                        actual_week = delta
                if isinstance(start.get("five_hour"), (int, float)) and isinstance(five, (int, float)):
                    delta = float(start["five_hour"]) - float(five)
                    if 0 < delta <= 50:
                        actual_five = delta
            except (ValueError, TypeError, StopIteration):
                pass
            connection.execute(
                "UPDATE reservations SET released_at = ?, release_reason = ?, "
                "actual_week = ?, actual_five = ? WHERE reservation_id = ?",
                (now, reason[:120], actual_week, actual_five, reservation_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO cost_observations "
                "(reservation_id, provider, size, q_week, q_five, success, observed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (reservation_id, row["provider"], row["size"], actual_week,
                 actual_five, 1 if success else 0, now),
            )
            provider_failure = reason == "terminal_provider_failed"
            # Ordinary task/test failures and host launch failures are not
            # provider-health evidence. Only a positively classified provider
            # failure advances the cooldown streak; a verified success clears it.
            if success or provider_failure:
                health = connection.execute(
                    "SELECT failure_streak FROM provider_health WHERE provider = ?",
                    (row["provider"],),
                ).fetchone()
                streak = (
                    0
                    if success
                    else int(health["failure_streak"] if health else 0) + 1
                )
                cooldown_until = None
                if provider_failure and streak >= 2:
                    cooldown_until = now + min(
                        21600, 1800 * (2 ** (streak - 2))
                    )
                connection.execute(
                    "INSERT INTO provider_health "
                    "(provider, failure_streak, cooldown_until, last_reason, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(provider) DO UPDATE SET failure_streak=excluded.failure_streak, "
                    "cooldown_until=excluded.cooldown_until, last_reason=excluded.last_reason, "
                    "updated_at=excluded.updated_at",
                    (row["provider"], streak, cooldown_until, reason[:120], now),
                )
            connection.commit()
            return True
        finally:
            connection.close()

    def renew(self, reservation_id: str, ttl_seconds: int = 1800) -> bool:
        """Extend a live run's lease from a bounded status observation."""
        if not reservation_id or not isinstance(ttl_seconds, int):
            return False
        if not 60 <= ttl_seconds <= 604800:
            return False
        now = float(self.now_fn())
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE reservations SET expires_at = ? "
                "WHERE reservation_id = ? AND released_at IS NULL",
                (now + ttl_seconds, reservation_id),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def active_reservations(self) -> list[dict[str, Any]]:
        now = float(self.now_fn())
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT reservation_id, provider, model, lane, size, q_week, q_five, "
                "created_at, expires_at FROM reservations "
                "WHERE released_at IS NULL AND expires_at > ? ORDER BY created_at",
                (now,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def diagnostics(self) -> dict[str, Any]:
        now = float(self.now_fn())
        connection = self._connect()
        try:
            health = [dict(row) for row in connection.execute(
                "SELECT provider, failure_streak, cooldown_until, last_reason, "
                "updated_at FROM provider_health ORDER BY provider"
            ).fetchall()]
            costs = [dict(row) for row in connection.execute(
                "SELECT provider, size, COUNT(*) AS samples, "
                "AVG(q_week) AS avg_week, AVG(q_five) AS avg_five "
                "FROM cost_observations WHERE success = 1 "
                "GROUP BY provider, size ORDER BY provider, size"
            ).fetchall()]
            return {
                "generated_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                "active_reservations": self.active_reservations(),
                "provider_health": health,
                "cost_calibration": costs,
            }
        finally:
            connection.close()

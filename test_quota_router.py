import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quota_router import QuotaRouter


NOW = datetime(2026, 7, 27, 17, 30, tzinfo=timezone.utc).timestamp()


class QuotaRouterTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.quota = self.root / "quota"
        self.quota.mkdir()
        self.gemini_marker = self.root / "gemini-worker-enabled"
        self.gemini_marker.touch()
        self.router = QuotaRouter(
            self.quota,
            self.root / "router.sqlite3",
            self.gemini_marker,
            now_fn=lambda: NOW,
        )

    def write_model_usage(
        self,
        *,
        claude_week=45,
        claude_five=3,
        gemini_week=2,
        gemini_five=0,
        generated="2026-07-27T17:29:00Z",
        claude_week_reset="2026-07-30T10:00:00Z",
        claude_five_reset="2026-07-27T17:50:00Z",
    ):
        (self.quota / "model-usage.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "generated_at": generated,
                    "claude": {
                        "ok": True,
                        "source": "claude internal usage",
                        "windows": {
                            "weekly": {
                                "used_percent": claude_week,
                                "resets_at": claude_week_reset,
                            },
                            "five_hour": {
                                "used_percent": claude_five,
                                "resets_at": claude_five_reset,
                            },
                        },
                    },
                    "gemini": {
                        "ok": True,
                        "source": "agy /usage",
                        "windows": {
                            "weekly": {
                                "used_percent": gemini_week,
                                "refreshes_in": "66h 30m",
                            },
                            "five_hour": {
                                "used_percent": gemini_five,
                                "refreshes_in": "1h 45m",
                            },
                        },
                    },
                }
            )
        )

    def write_codex(self, used=55, five=None, generated="2026-07-27T17:29:00Z"):
        limits = {
            "primary": {
                "usedPercent": used,
                "windowDurationMins": 10080,
                "resetsAt": int(NOW + 66 * 3600),
            },
            "secondary": None,
            "rateLimitReachedType": None,
            "spendControlReached": False,
        }
        if five is not None:
            limits["secondary"] = {
                "usedPercent": five,
                "windowDurationMins": 300,
                "resetsAt": int(NOW + 2 * 3600),
            }
        (self.quota / "worker1-codex.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "generated_at": generated,
                    "rateLimits": limits,
                }
            )
        )

    def test_live_worker_shape_prefers_green_gemini_and_marks_codex_green_weekly_only(
        self,
    ):
        self.write_model_usage()
        self.write_codex()
        result = self.router.recommend(lane="worker", size="small")
        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "gemini")
        states = {item["provider"]: item["state"] for item in result["candidates"]}
        self.assertEqual(states["codex"], "GREEN")
        self.assertEqual(states["claude"], "GREEN")
        self.assertEqual(states["gemini"], "GREEN")
        codex = next(x for x in result["candidates"] if x["provider"] == "codex")
        self.assertEqual(codex["reason"], "within_weekly_reserve")
        self.assertFalse(codex["five_hour_applicable"])

    def test_strategy_lane_is_rejected(self):
        self.write_model_usage()
        self.write_codex()
        self.assertEqual(
            self.router.recommend(lane="strategy", size="small"),
            {"ok": False, "error": "bad lane"},
        )

    def test_localworker_short_circuits_cloud_scoring(self):
        result = self.router.recommend(localworker_eligible=True)
        self.assertEqual(result["provider"], "localworker")
        self.assertEqual(result["score"], 100.0)

    def test_explicit_pin_obeys_availability_but_not_preference(self):
        self.write_model_usage()
        self.write_codex()
        result = self.router.recommend(
            lane="worker",
            size="small",
            explicit_provider="claude",
        )
        self.assertEqual(result["provider"], "claude")
        self.assertTrue(result["reason"].startswith("explicit_pin:"))

    def test_stale_provider_is_red(self):
        self.write_model_usage(generated="2026-07-27T16:00:00Z")
        self.write_codex(generated="2026-07-27T16:00:00Z")
        result = self.router.recommend()
        self.assertFalse(result["ok"])
        self.assertEqual(
            {item["state"] for item in result["candidates"]}, {"RED"}
        )

    def test_gemini_worker_is_red_without_headless_permission_marker(self):
        self.write_model_usage()
        self.write_codex()
        self.gemini_marker.unlink()
        result = self.router.recommend(lane="worker", size="small")
        gemini = next(
            item for item in result["candidates"] if item["provider"] == "gemini"
        )
        self.assertEqual(gemini["state"], "RED")
        self.assertEqual(gemini["reason"], "headless_permissions_unconfigured")
        self.assertEqual(result["provider"], "claude")

    def test_reservation_changes_next_atomic_choice(self):
        self.write_model_usage(
            claude_week=82,
            claude_five=60,
            gemini_week=83,
            gemini_five=60,
            claude_week_reset="2026-07-27T18:00:00Z",
            claude_five_reset="2026-07-27T18:00:00Z",
        )
        self.write_codex(used=99)
        providers = []
        barrier = threading.Barrier(2)

        def choose(identifier):
            barrier.wait()
            result = self.router.recommend(
                lane="worker",
                size="large",
                allowed_providers=("claude", "gemini"),
                reserve=True,
                reservation_id=identifier,
            )
            self.assertTrue(result["ok"], result)
            providers.append(result["provider"])

        first = threading.Thread(target=choose, args=("one",))
        second = threading.Thread(target=choose, args=("two",))
        first.start()
        second.start()
        first.join()
        second.join()
        self.assertEqual(sorted(providers), ["claude", "gemini"])
        self.assertEqual(len(self.router.active_reservations()), 2)

    def test_release_removes_active_reservation(self):
        self.write_model_usage()
        self.write_codex()
        result = self.router.recommend(
            reserve=True, reservation_id="release-me"
        )
        self.assertTrue(result["reserved"])
        self.assertTrue(self.router.release("release-me", "terminal_success"))
        self.assertEqual(self.router.active_reservations(), [])

    def test_two_terminal_failures_put_provider_on_cooldown(self):
        self.write_model_usage()
        self.write_codex()
        for identifier in ("fail-one", "fail-two"):
            result = self.router.recommend(
                explicit_provider="claude", reserve=True, reservation_id=identifier
            )
            self.assertTrue(result["ok"])
            self.assertTrue(
                self.router.release(identifier, "terminal_provider_failed")
            )
        result = self.router.recommend()
        claude = next(x for x in result["candidates"] if x["provider"] == "claude")
        self.assertEqual(claude["state"], "RED")
        self.assertEqual(claude["reason"], "provider_cooldown")

    def test_success_resets_provider_failure_streak(self):
        self.write_model_usage()
        self.write_codex()
        for identifier, reason in (
            ("failed", "terminal_provider_failed"),
            ("success", "terminal_success"),
        ):
            result = self.router.recommend(
                explicit_provider="claude", reserve=True, reservation_id=identifier
            )
            self.assertTrue(result["ok"])
            self.assertTrue(self.router.release(identifier, reason))
        health = self.router.diagnostics()["provider_health"]
        claude = next(x for x in health if x["provider"] == "claude")
        self.assertEqual(claude["failure_streak"], 0)
        self.assertIsNone(claude["cooldown_until"])

    def test_ordinary_task_failure_does_not_trip_provider_cooldown(self):
        self.write_model_usage()
        self.write_codex()
        for identifier in ("task-fail-one", "task-fail-two"):
            result = self.router.recommend(
                explicit_provider="claude", reserve=True, reservation_id=identifier
            )
            self.assertTrue(result["ok"])
            self.assertTrue(self.router.release(identifier, "terminal_failed"))
        result = self.router.recommend()
        claude = next(x for x in result["candidates"] if x["provider"] == "claude")
        self.assertNotEqual(claude["reason"], "provider_cooldown")

    def test_live_status_lease_can_be_renewed(self):
        self.write_model_usage()
        self.write_codex()
        result = self.router.recommend(
            reserve=True, reservation_id="renew-me", ttl_seconds=60
        )
        self.assertTrue(result["ok"])
        before = self.router.active_reservations()[0]["expires_at"]
        self.assertTrue(self.router.renew("renew-me", 600))
        after = self.router.active_reservations()[0]["expires_at"]
        self.assertGreater(after, before)

    def test_newest_failed_codex_snapshot_selects_older_fresh_success(self):
        self.write_model_usage()
        self.write_codex(used=40, generated="2026-07-27T17:28:00Z")
        (self.quota / "worker2-codex.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": "backend_timeout",
                    "generated_at": "2026-07-27T17:29:00Z",
                }
            )
        )
        telemetry = self.router._telemetry(NOW)
        self.assertTrue(telemetry["codex"]["ok"])
        self.assertEqual(telemetry["codex"]["weekly"]["used"], 40.0)

    def test_valid_weekly_only_codex_telemetry_is_green_within_weekly_reserve(self):
        self.write_model_usage()
        self.write_codex(used=50, five=None)
        result = self.router.recommend()
        codex = next(x for x in result["candidates"] if x["provider"] == "codex")
        self.assertEqual(codex["state"], "GREEN")
        self.assertEqual(codex["reason"], "within_weekly_reserve")
        self.assertFalse(codex["five_hour_applicable"])
        self.assertEqual(codex["confidence"], 1.0)
        self.assertEqual(codex["predicted_cost"]["five_hour"], 0.0)
        self.assertIsNone(codex["remaining"]["five_hour"])
        self.assertIsNone(codex["post_task"]["five_hour"])

    def test_codex_weekly_reserve_breach_is_amber(self):
        self.write_model_usage()
        self.write_codex(used=90, five=None)
        result = self.router.recommend()
        codex = next(x for x in result["candidates"] if x["provider"] == "codex")
        self.assertEqual(codex["state"], "AMBER")
        self.assertEqual(codex["reason"], "weekly_reserve_breach")

    def test_codex_insufficient_weekly_capacity_is_red(self):
        self.write_model_usage()
        self.write_codex(used=99, five=None)
        result = self.router.recommend()
        codex = next(x for x in result["candidates"] if x["provider"] == "codex")
        self.assertEqual(codex["state"], "RED")
        self.assertEqual(codex["reason"], "predicted_capacity_insufficient")

    def test_codex_provider_error_paths_remain_red(self):
        self.write_model_usage()
        (self.quota / "worker1-codex.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": "backend_timeout",
                    "generated_at": "2026-07-27T17:29:00Z",
                }
            )
        )
        result = self.router.recommend()
        codex = next(x for x in result["candidates"] if x["provider"] == "codex")
        self.assertEqual(codex["state"], "RED")
        self.assertEqual(codex["reason"], "telemetry_unavailable")

    def test_codex_five_hour_cost_and_reservations_are_always_zero(self):
        self.write_model_usage()
        self.write_codex(used=50, five=None)
        result = self.router.recommend(
            explicit_provider="codex", reserve=True, reservation_id="codex-cost"
        )
        self.assertTrue(result["ok"])
        codex = next(x for x in result["candidates"] if x["provider"] == "codex")
        self.assertEqual(codex["predicted_cost"]["five_hour"], 0.0)
        active = self.router.active_reservations()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["q_five"], 0.0)

    def test_claude_and_gemini_five_hour_semantics_unchanged(self):
        self.write_model_usage()
        self.write_codex()
        result = self.router.recommend(lane="worker", size="small")
        claude = next(x for x in result["candidates"] if x["provider"] == "claude")
        gemini = next(x for x in result["candidates"] if x["provider"] == "gemini")
        for candidate in (claude, gemini):
            self.assertTrue(candidate["five_hour_applicable"])
            self.assertIsNotNone(candidate["remaining"]["five_hour"])
            self.assertIsNotNone(candidate["post_task"]["five_hour"])
        self.assertEqual(claude["state"], "GREEN")
        self.assertEqual(claude["reason"], "within_dynamic_reserves")
        self.assertEqual(gemini["state"], "GREEN")
        self.assertEqual(gemini["reason"], "within_dynamic_reserves")

    def test_all_recent_codex_snapshots_failed_returns_telemetry_unavailable(self):
        self.write_model_usage()
        (self.quota / "worker1-codex.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": "backend_timeout",
                    "generated_at": "2026-07-27T17:29:00Z",
                }
            )
        )
        result = self.router.recommend()
        codex = next(x for x in result["candidates"] if x["provider"] == "codex")
        self.assertEqual(codex["state"], "RED")
        self.assertEqual(codex["reason"], "telemetry_unavailable")


if __name__ == "__main__":
    unittest.main()

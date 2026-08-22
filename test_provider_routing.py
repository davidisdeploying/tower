import json
import os
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import kicker


class ProviderRoutingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.quota = os.path.join(self.root, "quota")
        self.runs = os.path.join(self.root, "runs")
        self.router_db = os.path.join(self.root, "router.sqlite3")
        self.gemini_marker = os.path.join(self.root, "gemini-worker-enabled")
        os.makedirs(self.quota)
        os.makedirs(self.runs)
        open(self.gemini_marker, "a").close()
        self.write_model_usage()

    def write_model_usage(
        self,
        claude_week=45,
        claude_five=3,
        gemini_week=2,
        gemini_five=0,
    ):
        now = datetime.now(timezone.utc)
        with open(os.path.join(self.quota, "model-usage.json"), "w") as handle:
            json.dump(
                {
                    "ok": True,
                    "generated_at": kicker._now(),
                    "claude": {
                        "ok": True,
                        "source": "test",
                        "windows": {
                            "weekly": {
                                "used_percent": claude_week,
                                "resets_at": (now + timedelta(days=3)).isoformat(),
                            },
                            "five_hour": {
                                "used_percent": claude_five,
                                "resets_at": (now + timedelta(hours=2)).isoformat(),
                            },
                        },
                    },
                    "gemini": {
                        "ok": True,
                        "source": "test",
                        "windows": {
                            "weekly": {
                                "used_percent": gemini_week,
                                "refreshes_in": "72h",
                            },
                            "five_hour": {
                                "used_percent": gemini_five,
                                "refreshes_in": "2h",
                            },
                        },
                    },
                },
                handle,
            )

    def write_quota(self, seat="worker1", used=10):
        with open(os.path.join(self.quota, f"{seat}-codex.json"), "w") as handle:
            json.dump(
                {
                    "ok": True,
                    "generated_at": kicker._now(),
                    "rateLimits": {
                        "primary": {"usedPercent": used, "windowDurationMins": 10080},
                        "rateLimitReachedType": None,
                        "spendControlReached": False,
                    },
                },
                handle,
            )

    def test_explicit_codex_requires_fresh_available_quota(self):
        with mock.patch.object(kicker, "QUOTA_ROOT", self.quota), mock.patch.object(
            kicker, "QUOTA_ROUTER_DB", self.router_db
        ), mock.patch.object(
            kicker, "GEMINI_WORKER_MARKER", self.gemini_marker
        ):
            self.assertEqual(
                kicker._provider_spec("worker1", "codex", self.runs)["error"],
                "codex:telemetry_unavailable",
            )
            self.write_quota(used=100)
            self.assertEqual(
                kicker._provider_spec("worker1", "codex", self.runs)["error"],
                "codex:predicted_capacity_insufficient",
            )
            self.write_quota(used=10)
            spec = kicker._provider_spec("worker1", "codex", self.runs)
        self.assertTrue(spec["ok"])
        self.assertEqual(spec["provider"], "codex")
        self.assertEqual(spec["model"], "gpt-5.6-terra")

    def test_auto_uses_quota_router_across_all_three_providers(self):
        with mock.patch.object(kicker, "QUOTA_ROOT", self.quota), mock.patch.object(
            kicker, "QUOTA_ROUTER_DB", self.router_db
        ), mock.patch.object(
            kicker, "GEMINI_WORKER_MARKER", self.gemini_marker
        ):
            self.write_quota()
            spec = kicker._provider_spec("worker1", "auto", self.runs)
        self.assertEqual(spec["provider"], "gemini")
        self.assertEqual(spec["routing"]["state"], "GREEN")

    def test_codex_run_script_uses_isolated_worker_profile(self):
        token = "FLEET-WORKER1-RECON-20260727-script"
        runs = os.path.join(self.root, "from-worker1", "runs")
        transcripts = os.path.join(self.root, "from-worker1", "transcripts")
        response = SimpleNamespace(
            returncode=0, stdout="WORKER1_LAUNCHED\n", stderr=""
        )
        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", return_value=response
        ):
            result = kicker._kick_remote(
                token,
                "recon",
                "",
                seat="worker1",
                to_seat="to-worker1",
                from_seat="from-worker1",
                runs_root=runs,
                transcripts_dir=transcripts,
                claude_bin=kicker.CODEX_BIN,
                model=kicker.CODEX_WORKER_MODEL,
                host="worker1",
                host_ip=None,
                seat_env="relay-worker1",
                footer_fn=kicker._build_footer,
                marker="WORKER1_LAUNCHED",
                launched_note="test",
                prompt_content=f"{token}\n",
                provider="codex",
                provider_reason="explicit_codex",
            )
        self.assertTrue(result["ok"], result)
        with open(os.path.join(runs, token, "run.sh"), encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn("export CODEX_HOME=/home/david/.codex-worker", script)
        self.assertIn("exec --json", script)
        self.assertIn('"provider":"codex"', script)
        self.assertNotIn("--dangerously-skip-permissions", script)
        self.assertIn('provider == "codex"', script)
        self.assertIn("atomic_write(token_response, answer)", script)
        prompt = (Path(runs) / token / "prompt.txt").read_text(encoding="utf-8")
        self.assertIn("automated by the cloud-worker launcher", prompt)
        self.assertNotIn("APPEND that same response", prompt)

    def test_gemini_run_script_uses_worker_profile_and_stream_json(self):
        token = "FLEET-WORKER1-RECON-20260727-gemini-script"
        runs = os.path.join(self.root, "gemini-worker1", "runs")
        transcripts = os.path.join(self.root, "gemini-worker1", "transcripts")
        response = SimpleNamespace(returncode=0, stdout="WORKER1_LAUNCHED\n", stderr="")
        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", return_value=response
        ):
            result = kicker._kick_remote(
                token,
                "recon",
                "",
                seat="worker1",
                to_seat="to-worker1",
                from_seat="from-worker1",
                runs_root=runs,
                transcripts_dir=transcripts,
                claude_bin=kicker.AGY_BIN,
                model=kicker.GEMINI_WORKER_MODEL,
                host="worker1",
                host_ip=None,
                seat_env="relay-worker1",
                footer_fn=kicker._build_footer,
                marker="WORKER1_LAUNCHED",
                launched_note="test",
                prompt_content=f"{token}\n",
                provider="gemini",
                provider_reason="auto_green:test",
            )
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["transcript"].endswith(".json"))
        with open(os.path.join(runs, token, "run.sh"), encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn("--agent fleet-worker", script)
        self.assertIn("gemini-3.6-flash-high", script)
        self.assertIn('"provider":"gemini"', script)
        self.assertIn("--dangerously-skip-permissions", script)
        self.assertIn("--output-format stream-json", script)
        self.assertIn('record.get("event") == "result"', script)
        self.assertIn("atomic_write(token_response, answer)", script)
        self.assertNotIn('cp "$TR/', script)
        self.assertNotIn(f"$TR/{token}.txt", script)
        self.assertIn("responses.md", script)
        prompt = (Path(runs) / token / "prompt.txt").read_text(encoding="utf-8")
        self.assertIn("RELAY DELIVERY (automated by the cloud-worker launcher)", prompt)
        self.assertIn("Do NOT manually write response.md", prompt)
        self.assertNotIn("APPEND that same response", prompt)


if __name__ == "__main__":
    unittest.main()

import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import kicker
import server


class AtomicDispatchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.runs = os.path.join(self.root, "from-test", "runs")
        self.transcripts = os.path.join(self.root, "from-test", "transcripts")
        contract_dir = os.path.join(
            self.root, "conventions", "fleet-strategy-contract"
        )
        os.makedirs(contract_dir)
        self.artifact_contract = (
            "## Worker artifact invariant\n\nTEST ARTIFACT CONTRACT"
        )
        with open(os.path.join(contract_dir, "AGENTS.md"), "w") as handle:
            handle.write(
                kicker._WORKER_ARTIFACT_BEGIN
                + "\n"
                + self.artifact_contract
                + "\n"
                + kicker._WORKER_ARTIFACT_END
                + "\n"
            )

    @staticmethod
    def footer(*_args):
        return "\nFOOTER\n"

    @staticmethod
    def success(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="TEST_LAUNCHED\n", stderr="")

    def kick(self, token, content, **kwargs):
        return kicker._kick_remote(
            token, "prompts", "", seat="test", to_seat="to-test",
            from_seat="from-test", runs_root=self.runs,
            transcripts_dir=self.transcripts, claude_bin="/bin/true",
            model="test", host="test", host_ip="test-ip",
            seat_env="relay-test", footer_fn=self.footer,
            marker="TEST_LAUNCHED", launched_note="test",
            prompt_content=content,
            **kwargs,
        )

    def test_direct_prompt_needs_no_latest_and_is_preserved_exactly(self):
        token = "FLEET-TEST-BUILD-20260722-direct"
        content = f"{token}\nexact source\n{token}\n"
        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", side_effect=self.success
        ):
            result = self.kick(token, content)
        self.assertTrue(result["ok"], result)
        request = os.path.join(self.runs, token, "request.md")
        with open(request, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), content)
        self.assertEqual(stat.S_IMODE(os.stat(request).st_mode), 0o444)
        with open(os.path.join(self.runs, token, "prompt.txt"), encoding="utf-8") as handle:
            self.assertEqual(
                handle.read(),
                content.rstrip()
                + "\n\n---\n\n"
                + self.artifact_contract
                + "\n\nFOOTER\n",
            )
        self.assertFalse(os.path.exists(os.path.join(self.root, "to-test", "prompts", "latest.md")))

    def test_missing_contract_uses_bounded_fallback(self):
        with mock.patch.object(kicker, "VAULT", self.root):
            os.remove(
                os.path.join(
                    self.root,
                    "conventions",
                    "fleet-strategy-contract",
                    "AGENTS.md",
                )
            )
            contract = kicker._worker_artifact_contract()
        self.assertIn("Worker artifact invariant", contract)
        self.assertIn("files/YYYY/YYYY-MM-DD", contract)

    def test_worker_route_and_execution_context_are_launcher_owned(self):
        token = "FLEET-TEST-BUILD-20260722-worker-route"
        content = f"{token}\nexact caller source\n"
        route = {
            "seat": "test",
            "worker_host": "alpha",
            "target_host": "delta",
            "target_scope": "repo:/home/david/loupe",
            "cross_host": True,
        }
        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", side_effect=self.success
        ):
            result = self.kick(
                token,
                content,
                worker_routing=route,
                execution_context="Use `ssh delta`; authority remains bounded.",
            )
        self.assertTrue(result["ok"], result)
        run_dir = os.path.join(self.runs, token)
        with open(os.path.join(run_dir, "request.md"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), content)
        with open(os.path.join(run_dir, "worker-routing.json"), encoding="utf-8") as handle:
            self.assertEqual(__import__("json").load(handle), route)
        with open(os.path.join(run_dir, "prompt.txt"), encoding="utf-8") as handle:
            executable = handle.read()
        self.assertIn("ROUTED EXECUTION CONTEXT", executable)
        self.assertIn("Use `ssh delta`; authority remains bounded.", executable)

    def test_concurrent_different_content_same_token_has_one_immutable_winner(self):
        token = "FLEET-TEST-BUILD-20260722-race"
        barrier = threading.Barrier(2)
        bodies = [f"{token}\nbody-a\n", f"{token}\nbody-b\n"]

        def launch(content):
            barrier.wait()
            return self.kick(token, content)

        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", side_effect=self.success
        ) as run:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(launch, bodies))
        self.assertEqual(sum(bool(r["ok"]) for r in results), 1)
        self.assertEqual(run.call_count, 1)
        with open(os.path.join(self.runs, token, "request.md"), encoding="utf-8") as handle:
            self.assertIn(handle.read(), bodies)

    def test_invalid_content_and_missing_token_do_not_claim(self):
        token = "FLEET-TEST-BUILD-20260722-invalid"
        with mock.patch.object(kicker, "VAULT", self.root):
            invalid = self.kick(token, "  ")
            mismatch = self.kick(token, "different prompt")
        self.assertEqual(invalid["error"], "invalid prompt content")
        self.assertEqual(mismatch["error"], "token mismatch")
        self.assertFalse(os.path.exists(os.path.join(self.runs, token)))

    def test_dispatch_maps_all_seats_and_lanes(self):
        token = "FLEET-TEST-BUILD-20260722-map"
        content = f"{token}\nbody\n"
        names = ("delta", "charlie", "localworker", "worker4", "alpha")
        for seat in names:
            for lane in kicker.LANES:
                with self.subTest(seat=seat, lane=lane), mock.patch.object(
                    kicker, f"kick_{seat}", return_value={"ok": True}
                ) as launch:
                    result = server.dispatch(seat, lane, token, content, "session.md")
                    self.assertTrue(result["ok"])
                    expected = {
                        "lane": lane,
                        "session_note": "session.md",
                        "prompt_content": content,
                        "max_runtime_seconds": 0,
                    }
                    if seat in {"delta", "charlie", "alpha"}:
                        expected.update(provider="auto", task_size="small")
                    launch.assert_called_once_with(token, **expected)

    def test_dispatch_validation_precedes_launch(self):
        with mock.patch.object(kicker, "kick_alpha") as launch:
            self.assertEqual(server.dispatch("bad", "prompts", "x", "x")["error"], "bad seat")
            self.assertEqual(server.dispatch("worker2", "bad", "x", "x")["error"], "bad lane")
            self.assertEqual(server.dispatch("worker2", "prompts", "x", " ")["error"], "invalid prompt content")
            self.assertEqual(
                server.dispatch("worker2", "prompts", "x", "x", provider="bad")["error"],
                "bad provider",
            )
        launch.assert_not_called()

    def test_explicit_codex_provider_is_forwarded(self):
        token = "FLEET-WORKER2-RECON-20260727-codex"
        content = f"{token}\nbody\n"
        with mock.patch.object(kicker, "kick_alpha", return_value={"ok": True}) as launch:
            result = server.dispatch("worker2", "recon", token, content, provider="codex")
        self.assertTrue(result["ok"])
        launch.assert_called_once_with(
            token, lane="recon", session_note="", prompt_content=content,
            max_runtime_seconds=0, provider="codex", task_size="small",
        )

    def test_auto_localworker_first_requires_and_routes_compact_charlie_work(self):
        token = "FLEET-LOCALWORKER-RECON-20260729-auto-compact"
        content = "\n".join(
            (
                token,
                "Read the exact bounded source and report the required markers.",
                "FLEET_COMPACT_DELIVERY_V1_BEGIN",
                '{"tool_sequence":["read_file","finish_task_compact"],'
                '"required_response_markers":["## VERDICT","## Evidence",'
                f'"## Rollback","{token}"]}}',
                "FLEET_COMPACT_DELIVERY_V1_END",
                token,
            )
        )
        with mock.patch.object(
            kicker, "kick_localworker", return_value={"ok": True}
        ) as localworker, mock.patch.object(server, "_select_worker") as cloud:
            result = server.dispatch(
                "auto",
                "recon",
                token,
                content,
                target_host="charlie",
                localworker_eligible=True,
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["worker_routing"]["seat"], "localworker")
        self.assertEqual(
            result["worker_routing"]["reason"],
            "localworker_first_compact_contract",
        )
        cloud.assert_not_called()
        localworker.assert_called_once()

    def test_auto_localworker_first_fails_closed_outside_contract(self):
        token = "FLEET-LOCALWORKER-RECON-20260729-auto-invalid"
        content = f"{token}\nNo compact contract.\n{token}\n"
        cases = (
            (
                {"seat": "worker1", "target_host": "charlie"},
                "localworker eligibility requires auto seat",
            ),
            (
                {"seat": "auto", "target_host": "charlie", "provider": "claude"},
                "localworker eligibility requires auto provider",
            ),
            (
                {"seat": "auto", "target_host": "delta"},
                "localworker auto requires charlie target",
            ),
            (
                {"seat": "auto", "target_host": "charlie"},
                "localworker auto requires compact delivery contract",
            ),
        )
        for kwargs, error in cases:
            with self.subTest(error=error), mock.patch.object(
                kicker, "kick_localworker"
            ) as launch:
                result = server.dispatch(
                    lane="recon",
                    token=token,
                    content=content,
                    localworker_eligible=True,
                    **kwargs,
                )
            self.assertEqual(result["error"], error)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

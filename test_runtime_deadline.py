import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import kicker
import server


class RuntimeDeadlineTestCase(unittest.TestCase):
    def test_validation_accepts_zero_and_positive_but_rejects_bad_values(self):
        self.assertEqual(kicker._normalize_max_runtime(None), 0)
        self.assertEqual(kicker._normalize_max_runtime(0), 0)
        self.assertEqual(kicker._normalize_max_runtime(1), 1)
        self.assertEqual(kicker._normalize_max_runtime(604800), 604800)
        for bad in (-1, 604801, True, 1.5, "1.0", "bad"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                kicker._normalize_max_runtime(bad)

    def test_runner_times_out_process_group_and_writes_terminal_reason(self):
        with tempfile.TemporaryDirectory() as root:
            runner = os.path.join(root, "runner.py")
            terminal = os.path.join(root, "terminal.json")
            with open(runner, "w", encoding="utf-8") as handle:
                handle.write(kicker._DEADLINE_RUNNER)
            started = time.monotonic()
            result = subprocess.run([
                sys.executable, runner, "--max-runtime", "1", "--terminal", terminal,
                "--", sys.executable, "-c", "import time; time.sleep(30)",
            ], capture_output=True, text=True, timeout=8)
            elapsed = time.monotonic() - started
            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertLess(elapsed, 5)
            with open(terminal, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["reason"], "max_runtime_exceeded")
            self.assertEqual(payload["max_runtime_seconds"], 1)
            self.assertIn("timed_out_at", payload)

    def _kick(self, root, token, max_runtime):
        runs = os.path.join(root, "from-test", "runs")
        transcripts = os.path.join(root, "from-test", "transcripts")
        content = f"{token}\nbody\n"
        return kicker._kick_remote(
            token, "prompts", "", seat="test", to_seat="to-test",
            from_seat="from-test", runs_root=runs, transcripts_dir=transcripts,
            claude_bin="/bin/true", model="test", host="host", host_ip="ip",
            seat_env="relay-test", footer_fn=lambda *_args: "\nfooter\n",
            marker="TEST_LAUNCHED", launched_note="test", prompt_content=content,
            max_runtime_seconds=max_runtime,
        )

    def test_deadline_run_persists_runner_and_wraps_worker(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(kicker, "VAULT", root), mock.patch.object(
            kicker.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="TEST_LAUNCHED\n", stderr=""
            )
        ):
            token = "FLEET-TEST-BUILD-20260722-deadline"
            result = self._kick(root, token, 30)
            self.assertTrue(result["ok"], result)
            run_dir = os.path.join(root, "from-test", "runs", token)
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "deadline_runner.py")))
            with open(os.path.join(run_dir, "run.sh"), encoding="utf-8") as handle:
                script = handle.read()
            self.assertIn('--max-runtime "30"', script)
            self.assertIn('terminal.json', script)

    def test_unlimited_default_creates_no_runner_and_keeps_direct_branch(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(kicker, "VAULT", root), mock.patch.object(
            kicker.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="TEST_LAUNCHED\n", stderr=""
            )
        ):
            token = "FLEET-TEST-BUILD-20260722-unlimited"
            result = self._kick(root, token, 0)
            self.assertTrue(result["ok"], result)
            run_dir = os.path.join(root, "from-test", "runs", token)
            self.assertFalse(os.path.exists(os.path.join(run_dir, "deadline_runner.py")))
            with open(os.path.join(run_dir, "run.sh"), encoding="utf-8") as handle:
                self.assertIn('if [ "0" -gt 0 ]', handle.read())

    def test_all_generated_seat_scripts_pass_bash_syntax_check(self):
        variants = (
            ("worker1", "test", "relay-worker1"),
            ("worker3", "test", "relay-worker3"),
            ("localworker", None, "relay-localworker"),
            ("worker4", "test", "relay-worker4"),
            ("worker2", "test", "relay-worker2"),
        )
        with tempfile.TemporaryDirectory() as root, mock.patch.object(kicker, "VAULT", root), mock.patch.object(
            kicker.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="TEST_LAUNCHED\n", stderr=""
            )
        ):
            scripts = []
            for seat, model, seat_env in variants:
                token = f"FLEET-{seat.upper()}-BUILD-20260722-syntax"
                runs = os.path.join(root, f"from-{seat}", "runs")
                transcripts = os.path.join(root, f"from-{seat}", "transcripts")
                result = kicker._kick_remote(
                    token, "prompts", "", seat=seat, to_seat=f"to-{seat}",
                    from_seat=f"from-{seat}", runs_root=runs,
                    transcripts_dir=transcripts, claude_bin="/bin/true",
                    model=model, host="host", host_ip="ip", seat_env=seat_env,
                    footer_fn=lambda *_args: "\nfooter\n", marker="TEST_LAUNCHED",
                    launched_note="test", prompt_content=f"{token}\nbody\n",
                    max_runtime_seconds=30,
                )
                self.assertTrue(result["ok"], result)
                scripts.append(os.path.join(runs, token, "run.sh"))
            for script in scripts:
                checked = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, f"{script}: {checked.stderr}")

    def test_invalid_deadline_rejected_before_io_for_every_seat(self):
        entrypoints = (
            kicker.kick_delta, kicker.kick_charlie, kicker.kick_localworker,
            kicker.kick_worker4, kicker.kick_alpha,
        )
        token = "FLEET-WORKER2-BUILD-20260722-invalid-deadline"
        with mock.patch("builtins.open") as opened, mock.patch.object(kicker.os, "makedirs") as makedirs:
            for entrypoint in entrypoints:
                with self.subTest(entrypoint=entrypoint.__name__):
                    result = entrypoint(token, max_runtime_seconds=-1)
                    self.assertFalse(result["ok"])
                    self.assertIn("max_runtime_seconds", result["error"])
        opened.assert_not_called()
        makedirs.assert_not_called()

    def test_per_seat_launch_tools_are_gone_and_dispatch_propagates_deadline(self):
        """WORKER1-1 retired the ten per-seat tools; dispatch is the single launch path."""
        for action in ("summon", "ask"):
            for seat in ("worker1", "worker3", "worker2", "localworker", "worker4",
                         "delta", "charlie", "alpha"):
                self.assertFalse(
                    hasattr(server, f"{action}_{seat}"),
                    f"{action}_{seat} should have been removed by WORKER1-1",
                )
        token = "FLEET-TEST-BUILD-20260722-schema"
        with mock.patch.object(kicker, "kick_alpha", return_value={"ok": True}) as launch:
            result = server.dispatch(
                "alpha", "prompts", token, token + "\nbody\n" + token,
                max_runtime_seconds=90,
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(launch.call_args.kwargs["max_runtime_seconds"], 90)

    def test_dispatch_propagates_deadline(self):
        token = "FLEET-WORKER2-BUILD-20260722-dispatch-deadline"
        content = f"{token}\nbody\n"
        with mock.patch.object(kicker, "kick_alpha", return_value={"ok": True}) as launch:
            result = server.dispatch("worker2", "prompts", token, content, "session", 120)
        self.assertTrue(result["ok"])
        launch.assert_called_once_with(
            token, lane="prompts", session_note="session", prompt_content=content,
            max_runtime_seconds=120, provider="auto", task_size="small",
        )

    def test_status_surfaces_honest_terminal_reason(self):
        with tempfile.TemporaryDirectory() as root:
            token = "FLEET-TEST-BUILD-20260722-timed-out"
            runs = os.path.join(root, "runs")
            transcripts = os.path.join(root, "transcripts")
            run_dir = os.path.join(runs, token)
            os.makedirs(run_dir)
            os.makedirs(transcripts)
            with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as handle:
                json.dump({"lane": "prompts", "model": "test", "max_runtime_seconds": 5}, handle)
            with open(os.path.join(run_dir, "done"), "w", encoding="utf-8") as handle:
                handle.write("124\n")
            with open(os.path.join(run_dir, "terminal.json"), "w", encoding="utf-8") as handle:
                json.dump({"reason": "max_runtime_exceeded", "max_runtime_seconds": 5}, handle)
            result = kicker._remote_status(token, runs, transcripts, "from-test", "test")
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["exit_code"], 124)
            self.assertEqual(result["max_runtime_seconds"], 5)
            self.assertEqual(result["terminal_reason"], "max_runtime_exceeded")


if __name__ == "__main__":
    unittest.main()

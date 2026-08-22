import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import kicker


class SingleUseTokenTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = self.tempdir.name
        self.runs_root = os.path.join(self.root, "from-test", "runs")
        self.transcripts = os.path.join(self.root, "from-test", "transcripts")
        self.to_seat = "to-test"
        self.from_seat = "from-test"
        os.makedirs(os.path.join(self.root, self.to_seat, "prompts"))

    def _stage(self, token):
        path = os.path.join(self.root, self.to_seat, "prompts", "latest.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{token}\nscoped test prompt\n{token}\n")

    @staticmethod
    def _footer(*_args):
        return "\nfooter\n"

    def _kick(self, token, *, local=False):
        return kicker._kick_remote(
            token,
            "prompts",
            "",
            seat="test",
            to_seat=self.to_seat,
            from_seat=self.from_seat,
            runs_root=self.runs_root,
            transcripts_dir=self.transcripts,
            claude_bin="/bin/true",
            model="test-model",
            host="test-host",
            host_ip="test-host-ip",
            seat_env="relay-test",
            footer_fn=self._footer,
            marker="TEST_LAUNCHED",
            launched_note="test launch",
            local=local,
        )

    def _remote_success(self, *_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="TEST_LAUNCHED\n", stderr="")

    def test_concurrent_same_token_admits_exactly_one(self):
        token = "FLEET-TEST-BUILD-20260722-concurrent"
        self._stage(token)
        barrier = threading.Barrier(2)

        def launch():
            barrier.wait()
            return self._kick(token)

        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", side_effect=self._remote_success
        ) as run:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _n: launch(), range(2)))

        self.assertEqual([result["ok"] for result in results].count(True), 1)
        refused = [result for result in results if not result["ok"]]
        self.assertEqual(len(refused), 1)
        self.assertEqual(refused[0]["error"], "token already launched")
        self.assertEqual(run.call_count, 1)

    def test_existing_completed_token_is_refused_without_mutation(self):
        token = "FLEET-TEST-BUILD-20260722-completed"
        self._stage(token)
        run_dir = os.path.join(self.runs_root, token)
        os.makedirs(run_dir)
        done = os.path.join(run_dir, "done")
        with open(done, "w", encoding="utf-8") as handle:
            handle.write("0\n")
        before = os.stat(done).st_mtime_ns

        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run"
        ) as run:
            result = self._kick(token)

        self.assertEqual(result["error"], "token already launched")
        self.assertEqual(os.stat(done).st_mtime_ns, before)
        with open(done, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "0\n")
        run.assert_not_called()

    def test_existing_running_token_is_refused_without_mutation(self):
        token = "FLEET-TEST-RECON-20260722-running"
        self._stage(token)
        run_dir = os.path.join(self.runs_root, token)
        os.makedirs(run_dir)
        status = os.path.join(run_dir, "status.json")
        payload = {"token": token, "status": "running"}
        with open(status, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        before = os.stat(status).st_mtime_ns

        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run"
        ) as run:
            result = self._kick(token)

        self.assertEqual(result["error"], "token already launched")
        self.assertEqual(os.stat(status).st_mtime_ns, before)
        with open(status, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), payload)
        run.assert_not_called()

    def test_launch_failed_claim_remains_permanent(self):
        token = "FLEET-TEST-BUILD-20260722-launch-failed"
        self._stage(token)
        failed = SimpleNamespace(returncode=1, stdout="", stderr="boom")

        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", return_value=failed
        ):
            first = self._kick(token, local=True)
            second = self._kick(token, local=True)

        self.assertFalse(first["ok"])
        self.assertEqual(first["error"], "local launch failed")
        self.assertEqual(second["error"], "token already launched")
        run_dir = os.path.join(self.runs_root, token)
        with open(os.path.join(run_dir, "done"), "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "70\n")
        self.assertTrue(os.path.exists(os.path.join(run_dir, "launch-failed.txt")))

    def test_distinct_tokens_are_both_admitted(self):
        tokens = [
            "FLEET-TEST-BUILD-20260722-distinct-a",
            "FLEET-TEST-BUILD-20260722-distinct-b",
        ]
        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", side_effect=self._remote_success
        ):
            results = []
            for token in tokens:
                self._stage(token)
                results.append(self._kick(token))

        self.assertTrue(all(result["ok"] for result in results))
        self.assertTrue(all(os.path.isdir(os.path.join(self.runs_root, token)) for token in tokens))

    def test_local_launch_uses_the_same_claim(self):
        token = "FLEET-TEST-BUILD-20260722-local"
        self._stage(token)

        def local_success(argv, **_kwargs):
            run_sh = argv[-1]
            run_dir = os.path.dirname(run_sh)
            with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as handle:
                json.dump({"token": token, "status": "running"}, handle)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(kicker, "VAULT", self.root), mock.patch.object(
            kicker.subprocess, "run", side_effect=local_success
        ) as run:
            first = self._kick(token, local=True)
            second = self._kick(token, local=True)

        self.assertTrue(first["ok"])
        self.assertEqual(second["error"], "token already launched")
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()

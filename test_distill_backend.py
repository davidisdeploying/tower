import json
import os
import subprocess
import unittest
from unittest import mock

import server


class DistillBackendTestCase(unittest.TestCase):
    def test_live_default_is_opt_in(self):
        self.assertNotIn("DISTILL_BACKEND_CMD", os.environ)
        self.assertEqual(server._DISTILL_BACKEND_CMD, "")

    def test_empty_backend_uses_no_subprocess(self):
        with mock.patch.object(server, "_DISTILL_BACKEND_CMD", ""), mock.patch.object(
            server.subprocess, "run"
        ) as run:
            self.assertIsNone(server._distill_backend("summarize", {"text": "x"}))
        run.assert_not_called()

    @staticmethod
    def _completed(stdout_bytes=b"", stderr_bytes=b"", returncode=0):
        def fake_run(argv, **kwargs):
            kwargs["stdout"].write(stdout_bytes)
            kwargs["stderr"].write(stderr_bytes)
            return subprocess.CompletedProcess(argv, returncode)

        return fake_run

    def test_explicit_backend_success(self):
        payload = {"ok": True, "summary": "remote summary"}
        fake = self._completed(json.dumps(payload).encode("utf-8"))
        with mock.patch.object(server, "_DISTILL_BACKEND_CMD", "fake-backend --json"), mock.patch.object(
            server.subprocess, "run", side_effect=fake
        ) as run:
            result = server._distill_backend("summarize", {"text": "hello"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "command")
        self.assertEqual(result["operation"], "summarize")
        self.assertEqual(result["summary"], "remote summary")
        self.assertEqual(run.call_args.args[0], ["fake-backend", "--json", "summarize"])
        self.assertIsInstance(run.call_args.kwargs["input"], bytes)

    def test_nonzero_backend_error_is_bounded(self):
        fake = self._completed(stderr_bytes=b"x" * 10000, returncode=7)
        with mock.patch.object(server, "_DISTILL_BACKEND_CMD", "fake-backend"), mock.patch.object(
            server.subprocess, "run", side_effect=fake
        ):
            result = server._distill_backend("compress", {"text": "hello"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "backend failed")
        self.assertEqual(result["returncode"], 7)
        self.assertLessEqual(len(result["stderr"]), 1200)

    def test_backend_stdout_limit_precedes_json_parse(self):
        fake = self._completed(stdout_bytes=b"x" * (server._DISTILL_MAX_STDOUT_BYTES + 1))
        with mock.patch.object(server, "_DISTILL_BACKEND_CMD", "fake-backend"), mock.patch.object(
            server.subprocess, "run", side_effect=fake
        ):
            result = server._distill_backend("extract", {"text": "hello"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "backend output exceeded limit")
        self.assertEqual(result["stdout_bytes"], server._DISTILL_MAX_STDOUT_BYTES + 1)
        self.assertNotIn("stdout", result)

    def test_invalid_json_is_a_bounded_error(self):
        fake = self._completed(stdout_bytes=b"not-json")
        with mock.patch.object(server, "_DISTILL_BACKEND_CMD", "fake-backend"), mock.patch.object(
            server.subprocess, "run", side_effect=fake
        ):
            result = server._distill_backend("route", {"text": "hello"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "backend output was not valid JSON")
        self.assertEqual(result["stdout"], "not-json")

    def test_all_tools_fallback_when_backend_is_not_configured(self):
        with mock.patch.object(server, "_DISTILL_BACKEND_CMD", ""), mock.patch.object(
            server.subprocess, "run"
        ) as run:
            results = (
                server.distill_summarize("First sentence. Second sentence."),
                server.distill_compress("one\none\ntwo"),
                server.distill_route("fix a service with systemd"),
                server.distill_extract("error: failed\n/path/to/file"),
            )

        for result in results:
            self.assertTrue(result["ok"])
            self.assertEqual(result["backend"], "fallback")
            self.assertEqual(result["fallback_reason"], "backend_not_configured")
        run.assert_not_called()

    def test_all_tools_fallback_on_backend_error_without_payload_leak(self):
        error = {
            "ok": False,
            "backend": "command",
            "error": "backend failed",
            "returncode": 9,
            "stderr": "SECRET-PAYLOAD",
        }
        with mock.patch.object(server, "_distill_backend", return_value=error):
            results = (
                server.distill_summarize("secret summary input"),
                server.distill_compress("secret compression input"),
                server.distill_route("secret route input"),
                server.distill_extract("secret extract input"),
            )

        for result in results:
            self.assertTrue(result["ok"])
            self.assertEqual(result["backend"], "fallback")
            self.assertEqual(result["fallback_reason"], "backend_error")
            self.assertEqual(
                result["backend_error"],
                {"error": "backend failed", "returncode": 9},
            )
            self.assertNotIn("SECRET-PAYLOAD", repr(result))

    def test_successful_backend_result_wins(self):
        remote = {
            "ok": True,
            "backend": "command",
            "operation": "route",
            "decision": "remote",
        }
        with mock.patch.object(server, "_distill_backend", return_value=remote):
            self.assertIs(server.distill_route("anything"), remote)


if __name__ == "__main__":
    unittest.main()

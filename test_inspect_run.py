import json
import os
import tempfile
import unittest
from unittest import mock

import kicker
import server


class InspectRunTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.runs = os.path.join(self.root, "from-worker2", "runs")
        self.transcripts = os.path.join(self.root, "from-worker2", "transcripts")
        os.makedirs(self.runs)
        os.makedirs(self.transcripts)
        self.token = "FLEET-WORKER2-BUILD-20260722-inspect"
        self.run_dir = os.path.join(self.runs, self.token)
        os.makedirs(self.run_dir)
        with open(os.path.join(self.run_dir, "status.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "token": self.token, "lane": "prompts", "status": "running",
                "started_at": "2026-07-22T17:00:00Z", "model": "sonnet",
            }, handle)
        with open(os.path.join(self.run_dir, "done"), "w", encoding="utf-8") as handle:
            handle.write("0\n")
        response = "## VERDICT\nproof\n" + ("middle\n" * 500) + self.token + "\n"
        with open(os.path.join(self.run_dir, "response.md"), "w", encoding="utf-8") as handle:
            handle.write(response)

        transcript = os.path.join(self.transcripts, f"{self.token}.json")
        with open(transcript, "w", encoding="utf-8") as handle:
            for index in range(250):
                handle.write(json.dumps({
                    "type": "assistant", "timestamp": f"t{index}",
                    "message": {"role": "assistant", "content": [{
                        "type": "text", "text": "noise-" + ("x" * 500)
                    }]},
                }) + "\n")
            handle.write(json.dumps({
                "type": "assistant", "message": {"role": "assistant", "content": [{
                    "type": "tool_use", "id": "tool-final", "name": "Bash",
                    "input": {"command": "printf verified"},
                }]},
            }) + "\n")
            handle.write(json.dumps({
                "type": "user", "message": {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "tool-final",
                    "content": "verified-output", "is_error": False,
                }]},
            }) + "\n")

        self.patchers = [
            mock.patch.object(kicker, "VAULT", self.root),
            mock.patch.object(kicker, "DELTA_RUNS", os.path.join(self.root, "missing-worker1")),
            mock.patch.object(kicker, "CHARLIE_RUNS", os.path.join(self.root, "missing-worker3")),
            mock.patch.object(kicker, "LOCALWORKER_RUNS", os.path.join(self.root, "missing-localworker")),
            mock.patch.object(kicker, "WORKER4_RUNS", os.path.join(self.root, "missing-worker4")),
            mock.patch.object(kicker, "ALPHA_RUNS", self.runs),
            mock.patch.object(kicker, "ALPHA_TRANSCRIPTS", self.transcripts),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_complete_bounded_inspection(self):
        result = kicker.inspect_run(self.token, max_events=8, max_bytes=4096)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["seat"], "alpha")
        self.assertEqual(result["status"]["status"], "done")
        self.assertTrue(result["response"]["durable_per_token"])
        self.assertTrue(result["response"]["token_match"])
        self.assertTrue(result["response"]["truncated"])
        self.assertIn("## VERDICT", result["response"]["text"])
        self.assertIn(self.token, result["response"]["text"])
        self.assertLessEqual(result["transcript_tail"]["scanned_bytes"], 4096)
        self.assertTrue(result["transcript_tail"]["truncated"])
        self.assertEqual(result["tool_results"]["selected_count"], 1)
        self.assertFalse(result["tool_results"]["selection_complete"])
        evidence = result["tool_results"]["results"][0]
        self.assertEqual(evidence["tool_name"], "Bash")
        self.assertEqual(evidence["content"], "verified-output")
        self.assertLessEqual(result["output_text_bytes"], 4096)

    def test_include_is_selective_and_limits_are_clamped(self):
        result = kicker.inspect_run(
            self.token, include=["status"], max_events=999, max_bytes=1
        )
        self.assertEqual(set(result), {"ok", "token", "seat", "included", "limits", "status", "output_text_bytes", "output_text_truncated"})
        self.assertEqual(result["limits"]["max_events"], 100)
        self.assertEqual(result["limits"]["max_output_text_bytes"], 1024)
        self.assertNotIn("response", result)

    def test_bad_inputs_do_not_touch_files(self):
        self.assertEqual(kicker.inspect_run("bad/token")["error"], "malformed token")
        self.assertEqual(kicker.inspect_run(self.token, include=["secret"])["error"], "bad include")
        self.assertEqual(kicker.inspect_run("FLEET-WORKER2-BUILD-20260722-missing")["error"], "run not found")

    def test_ambiguous_token_is_reported(self):
        worker1_runs = os.path.join(self.root, "from-worker1", "runs")
        os.makedirs(os.path.join(worker1_runs, self.token))
        with mock.patch.object(kicker, "DELTA_RUNS", worker1_runs):
            result = kicker.inspect_run(self.token)
        self.assertEqual(result["error"], "ambiguous token")
        self.assertEqual(result["seats"], ["delta", "alpha"])

    def test_missing_transcript_and_response_are_explicit(self):
        os.remove(os.path.join(self.run_dir, "response.md"))
        os.remove(os.path.join(self.transcripts, f"{self.token}.json"))
        result = kicker.inspect_run(self.token)
        self.assertFalse(result["response"]["available"])
        self.assertFalse(result["transcript_tail"]["available"])
        self.assertFalse(result["tool_results"]["transcript_available"])

    def test_server_wrapper_preserves_arguments(self):
        with mock.patch.object(kicker, "inspect_run", return_value={"ok": True}) as inspect:
            result = server.inspect_run(self.token, ["status"], 3, 2048)
        self.assertTrue(result["ok"])
        inspect.assert_called_once_with(self.token, ["status"], 3, 2048)


if __name__ == "__main__":
    unittest.main()

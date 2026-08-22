import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import anyio
import jwt
from jwt import PyJWKClient

import kicker
import server


class _SlowJWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(0.25)
        try:
            body = b'{"keys": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format, *_args):
        pass


class RealisticJWKSTimeoutTestCase(unittest.TestCase):
    def test_real_socket_timeout_fails_closed_without_token_leak(self):
        httpd = HTTPServer(("127.0.0.1", 0), _SlowJWKSHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        token = jwt.encode(
            {"aud": "aud-one", "iss": server.CF_TEAM},
            "test-secret-key-that-is-at-least-32-bytes-long",
            algorithm="HS256",
            headers={"kid": "timeout-key"},
        )
        downstream_called = []
        events = []

        async def downstream(_scope, _receive, _send):
            downstream_called.append(True)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(event):
            events.append(event)

        middleware = server.CfAccessJWTMiddleware(downstream)
        middleware._jwks = PyJWKClient(
            f"http://127.0.0.1:{httpd.server_port}/jwks", timeout=0.05
        )
        scope = {
            "type": "http",
            "client": ("127.0.0.1", 12345),
            "headers": [
                (b"cf-ray", b"timeout-test"),
                (b"cf-access-jwt-assertion", token.encode("ascii")),
            ],
        }
        original_aud = server.CF_AUD
        started = time.monotonic()
        try:
            server.CF_AUD = ["aud-one"]
            with self.assertLogs("tower.gate", level="INFO") as captured:
                anyio.run(middleware, scope, receive, send)
        finally:
            server.CF_AUD = original_aud
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)
        elapsed = time.monotonic() - started

        self.assertFalse(downstream_called)
        self.assertEqual(events[0]["status"], 403)
        self.assertEqual(events[-1]["body"], b'{"error":"access denied"}')
        self.assertLess(elapsed, 1.5)
        joined_logs = "\n".join(captured.output)
        self.assertIn("PyJWKClientConnectionError", joined_logs)
        self.assertNotIn(token, joined_logs)


class LargeScaleRelayAuditTestCase(unittest.TestCase):
    def test_thousand_run_root_classifies_only_recent_window(self):
        with tempfile.TemporaryDirectory() as root:
            runs = os.path.join(root, "from-test", "runs")
            os.makedirs(runs)
            base = time.time() - 2000
            for index in range(1000):
                run_dir = os.path.join(runs, f"TOKEN-{index:04d}")
                os.makedirs(run_dir)
                stamp = base + index
                os.utime(run_dir, (stamp, stamp))

            classified = []

            def fake_run_entry(_run_dir, token, _source):
                classified.append(token)
                return {
                    "token": token,
                    "status": "running",
                    "started_at": None,
                    "model": "test",
                }

            with mock.patch.object(kicker, "VAULT", root), mock.patch.object(
                kicker, "_run_entry", side_effect=fake_run_entry
            ):
                result = kicker._seat_audit("test", runs, "from-test", 5)

        self.assertEqual(result["total_run_count"], 1000)
        self.assertEqual(result["inspected_run_count"], 5)
        self.assertEqual(result["inspection_limit"], 5)
        self.assertEqual(result["status_counts_scope"], "inspected_recent_runs_only")
        self.assertEqual(result["status_counts"]["running"], 5)
        self.assertEqual(len(set(classified)), 5)
        self.assertEqual(result["most_recent_run"]["token"], "TOKEN-0999")


if __name__ == "__main__":
    unittest.main()

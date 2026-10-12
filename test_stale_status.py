"""Focused tests for the stale-run classification added to kicker.py
(FLEET-WORKER2-BUILD-20260710-relay-stale-status).

Uses tempfile run dirs with controlled mtimes (os.utime) — no live processes,
no network, no /proc, no NAS. Run with:
    python -m unittest test_stale_status -v
"""
import json
import os
import shutil
import tempfile
import time
import unittest

import kicker


def _touch(path: str, age_seconds: float = 0.0, content: str = ""):
    """Create `path` with content, then backdate its mtime by age_seconds."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    if age_seconds:
        ts = time.time() - age_seconds
        os.utime(path, (ts, ts))


def _touch_dir(path: str, age_seconds: float = 0.0):
    os.makedirs(path, exist_ok=True)
    if age_seconds:
        ts = time.time() - age_seconds
        os.utime(path, (ts, ts))


class StaleStatusTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fleet-stale-test-")
        self.runs_root = os.path.join(self.tmp, "runs")
        self.transcripts = os.path.join(self.tmp, "transcripts")
        os.makedirs(self.runs_root)
        os.makedirs(self.transcripts)
        self._saved_env = {}
        for k in ("TOWER_STALE_AFTER_SECONDS", "TOWER_LAUNCH_GRACE_SECONDS"):
            self._saved_env[k] = os.environ.pop(k, None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _status_json(self, run_dir, status="running", extra=None):
        st = {
            "token": "TOK",
            "lane": "prompts",
            "pid": 12345,
            "started_at": "2026-07-10T00:00:00Z",
            "status": status,
            "transcript": os.path.join(self.transcripts, "TOK.json"),
            "response_path": "/tmp/resp.md",
            "source": "test",
        }
        if extra:
            st.update(extra)
        with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(st, f)

    # --- 1. done remains done (authoritative, no reclassification) ---------

    def test_done_remains_done(self):
        run_dir = os.path.join(self.runs_root, "TOK")
        os.makedirs(run_dir)
        self._status_json(run_dir)
        # Backdate heartbeat far past the stale threshold; done must still win.
        _touch(os.path.join(run_dir, "heartbeat"), age_seconds=99999)
        _touch(os.path.join(run_dir, "done"), content="0\n")
        result = kicker._remote_status("TOK", self.runs_root, self.transcripts, "from-test", "test")
        self.assertEqual(result["status"], "done")
        self.assertNotIn("stale_reason", result)
        self.assertEqual(result["exit_code"], 0)

    # --- 2. fresh heartbeat -> running ---------------------------------------

    def test_fresh_heartbeat_is_running(self):
        run_dir = os.path.join(self.runs_root, "TOK")
        os.makedirs(run_dir)
        self._status_json(run_dir)
        _touch(os.path.join(run_dir, "heartbeat"), age_seconds=5)
        result = kicker._remote_status("TOK", self.runs_root, self.transcripts, "from-test", "test")
        self.assertEqual(result["status"], "running")
        self.assertNotIn("stale_reason", result)
        self.assertIn("age_seconds", result)
        self.assertLess(result["age_seconds"], 30)

    # --- 3. old heartbeat -> stale/heartbeat_timeout -------------------------

    def test_old_heartbeat_is_stale(self):
        os.environ["TOWER_STALE_AFTER_SECONDS"] = "60"
        run_dir = os.path.join(self.runs_root, "TOK")
        os.makedirs(run_dir)
        self._status_json(run_dir)
        _touch(os.path.join(run_dir, "heartbeat"), age_seconds=120)
        result = kicker._remote_status("TOK", self.runs_root, self.transcripts, "from-test", "test")
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["stale_reason"], "heartbeat_timeout")
        self.assertGreaterEqual(result["age_seconds"], 100)
        self.assertIn("last_seen_at", result)
        # existing fields preserved
        self.assertEqual(result["source"], "test")
        self.assertEqual(result["started_at"], "2026-07-10T00:00:00Z")

    # --- 4. missing status.json, inside launch grace -> launching -----------

    def test_missing_status_inside_launch_grace_is_launching(self):
        os.environ["TOWER_LAUNCH_GRACE_SECONDS"] = "600"
        run_dir = os.path.join(self.runs_root, "TOK")
        _touch_dir(run_dir, age_seconds=10)
        result = kicker._remote_status("TOK", self.runs_root, self.transcripts, "from-test", "test")
        self.assertEqual(result["status"], "launching")
        self.assertNotIn("stale_reason", result)

    # --- 5. missing status.json, outside launch grace -> stale/launch_timeout

    def test_missing_status_outside_launch_grace_is_stale(self):
        os.environ["TOWER_LAUNCH_GRACE_SECONDS"] = "60"
        run_dir = os.path.join(self.runs_root, "TOK")
        _touch_dir(run_dir, age_seconds=120)
        result = kicker._remote_status("TOK", self.runs_root, self.transcripts, "from-test", "test")
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["stale_reason"], "launch_timeout")

    # --- 6. invalid threshold env values fall back to defaults --------------

    def test_invalid_threshold_env_falls_back(self):
        for bad in ("not-a-number", "-5", "0", "   ", "3.5"):
            os.environ["TOWER_STALE_AFTER_SECONDS"] = bad
            os.environ["TOWER_LAUNCH_GRACE_SECONDS"] = bad
            self.assertEqual(kicker._stale_after_seconds(), kicker._DEFAULT_STALE_AFTER_SECONDS)
            self.assertEqual(kicker._launch_grace_seconds(), kicker._DEFAULT_LAUNCH_GRACE_SECONDS)

    def test_unset_threshold_env_uses_default(self):
        self.assertEqual(kicker._stale_after_seconds(), 7200)
        self.assertEqual(kicker._launch_grace_seconds(), 600)

    # --- 7. Worker2-local path + remote-seat path ------------------------------

    def test_alpha_local_path_classification(self):
        orig_runs, orig_tr = kicker.ALPHA_RUNS, kicker.ALPHA_TRANSCRIPTS
        kicker.ALPHA_RUNS = self.runs_root
        kicker.ALPHA_TRANSCRIPTS = self.transcripts
        try:
            os.environ["TOWER_STALE_AFTER_SECONDS"] = "60"
            run_dir = os.path.join(self.runs_root, "TOK")
            os.makedirs(run_dir)
            self._status_json(run_dir)
            _touch(os.path.join(run_dir, "heartbeat"), age_seconds=300)
            result = kicker._worker2_status("TOK")
            self.assertEqual(result["status"], "stale")
            self.assertEqual(result["stale_reason"], "heartbeat_timeout")
            self.assertEqual(result["source"], "alpha")
        finally:
            kicker.ALPHA_RUNS, kicker.ALPHA_TRANSCRIPTS = orig_runs, orig_tr

    def test_charlie_remote_path_classification(self):
        orig_runs, orig_tr = kicker.CHARLIE_RUNS, kicker.CHARLIE_TRANSCRIPTS
        kicker.CHARLIE_RUNS = self.runs_root
        kicker.CHARLIE_TRANSCRIPTS = self.transcripts
        try:
            os.environ["TOWER_STALE_AFTER_SECONDS"] = "60"
            run_dir = os.path.join(self.runs_root, "TOK")
            os.makedirs(run_dir)
            self._status_json(run_dir)
            _touch(os.path.join(run_dir, "heartbeat"), age_seconds=5)
            result = kicker._worker3_status("TOK")
            self.assertEqual(result["status"], "running")
            self.assertEqual(result["source"], "charlie")
            # No /proc probe against a remote pid — confirm pid isn't consulted
            # for liveness (status came from heartbeat mtime only).
            self.assertNotIn("pid", result)
        finally:
            kicker.CHARLIE_RUNS, kicker.CHARLIE_TRANSCRIPTS = orig_runs, orig_tr

    # --- 8. no-token listing includes stale status ---------------------------

    def test_no_token_listing_includes_stale(self):
        os.environ["TOWER_STALE_AFTER_SECONDS"] = "60"
        os.environ["TOWER_LAUNCH_GRACE_SECONDS"] = "60"

        fresh_dir = os.path.join(self.runs_root, "FRESH")
        os.makedirs(fresh_dir)
        self._status_json(fresh_dir)
        _touch(os.path.join(fresh_dir, "heartbeat"), age_seconds=5)

        stale_dir = os.path.join(self.runs_root, "STALE")
        os.makedirs(stale_dir)
        self._status_json(stale_dir)
        _touch(os.path.join(stale_dir, "heartbeat"), age_seconds=999)

        done_dir = os.path.join(self.runs_root, "DONE")
        os.makedirs(done_dir)
        self._status_json(done_dir)
        _touch(os.path.join(done_dir, "heartbeat"), age_seconds=999)
        _touch(os.path.join(done_dir, "done"), content="0\n")

        launch_stale_dir = os.path.join(self.runs_root, "LAUNCHSTALE")
        _touch_dir(launch_stale_dir, age_seconds=999)

        entries = {e["token"]: e for e in kicker._list_runs_in(self.runs_root, "test")}

        self.assertEqual(entries["FRESH"]["status"], "running")
        self.assertEqual(entries["STALE"]["status"], "stale")
        self.assertEqual(entries["STALE"]["stale_reason"], "heartbeat_timeout")
        self.assertEqual(entries["DONE"]["status"], "done")
        self.assertNotIn("stale_reason", entries["DONE"])
        self.assertEqual(entries["LAUNCHSTALE"]["status"], "stale")
        self.assertEqual(entries["LAUNCHSTALE"]["stale_reason"], "launch_timeout")

    def test_status_no_token_lists_across_seats(self):
        # Full status() no-token path, monkeypatched onto our tmp roots.
        orig = {
            "WORKER1_RUNS": kicker.DELTA_RUNS, "WORKER3_RUNS": kicker.CHARLIE_RUNS,
            "LOCALWORKER_RUNS": kicker.LOCALWORKER_RUNS, "WORKER4_RUNS": kicker.WORKER4_RUNS,
            "WORKER2_RUNS": kicker.ALPHA_RUNS,
        }
        empty_dir = os.path.join(self.tmp, "empty_runs")
        kicker.DELTA_RUNS = empty_dir
        kicker.CHARLIE_RUNS = empty_dir
        kicker.LOCALWORKER_RUNS = empty_dir
        kicker.WORKER4_RUNS = empty_dir
        kicker.ALPHA_RUNS = self.runs_root
        # WORKER1-1: status() also scans the legacy seat roots so historical runs stay
        # listed. Point them at the empty dir or this test sees the real vault.
        legacy_orig = {n: getattr(kicker, n) for n in ("WORKER1_RUNS", "WORKER3_RUNS", "WORKER2_RUNS")}
        for name in legacy_orig:
            setattr(kicker, name, empty_dir)
        try:
            os.environ["TOWER_STALE_AFTER_SECONDS"] = "60"
            run_dir = os.path.join(self.runs_root, "TOK")
            os.makedirs(run_dir)
            self._status_json(run_dir)
            _touch(os.path.join(run_dir, "heartbeat"), age_seconds=999)
            result = kicker.status(None)
            self.assertTrue(result["ok"])
            alpha_runs = [r for r in result["runs"] if r["source"] == "alpha"]
            self.assertEqual(len(alpha_runs), 1)
            self.assertEqual(alpha_runs[0]["status"], "stale")
            self.assertEqual(alpha_runs[0]["stale_reason"], "heartbeat_timeout")
        finally:
            for k, v in orig.items():
                setattr(kicker, k, v)


if __name__ == "__main__":
    unittest.main()

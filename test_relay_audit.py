"""Focused tests for kicker.relay_audit (FLEET-WORKER2-BUILD-20260710-relay-audit).

Uses tempfile run dirs with controlled mtimes (os.utime), monkeypatched onto
kicker's seat-root globals — no live processes, no network, no NAS, no /proc.
Run with:
    python -m unittest test_relay_audit -v
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import kicker


def _touch(path: str, age_seconds: float = 0.0, content: str = ""):
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


def _status_json(run_dir, status="running", extra=None):
    st = {
        "token": os.path.basename(run_dir),
        "lane": "prompts",
        "pid": 12345,
        "started_at": "2026-07-10T00:00:00Z",
        "status": status,
        "response_path": "/tmp/resp.md",
        "source": "test",
    }
    if extra:
        st.update(extra)
    with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(st, f)


class RelayAuditTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fleet-relay-audit-test-")
        self._saved_env = {}
        for k in ("TOWER_STALE_AFTER_SECONDS", "TOWER_LAUNCH_GRACE_SECONDS"):
            self._saved_env[k] = os.environ.pop(k, None)

        # Snapshot every global relay_audit touches so each test can freely
        # rebind seat roots without leaking state into other tests.
        self._orig = {
            name: getattr(kicker, name)
            for name in ("WORKER1_RUNS", "WORKER3_RUNS", "LOCALWORKER_RUNS", "WORKER4_RUNS", "WORKER2_RUNS", "VAULT")
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for name, val in self._orig.items():
            setattr(kicker, name, val)

    def _seat_root(self, seat: str) -> str:
        root = os.path.join(self.tmp, f"from-{seat}", "runs")
        os.makedirs(root, exist_ok=True)
        return root

    def _bind_all_seats(self, roots: dict):
        """roots: {seat: run_root_path}; unset seats point at an empty dir."""
        empty = os.path.join(self.tmp, "empty_runs")
        os.makedirs(empty, exist_ok=True)
        for seat, attr in (
            ("worker1", "WORKER1_RUNS"), ("worker3", "WORKER3_RUNS"), ("localworker", "LOCALWORKER_RUNS"),
            ("worker4", "WORKER4_RUNS"), ("worker2", "WORKER2_RUNS"),
        ):
            setattr(kicker, attr, roots.get(seat, empty))
        kicker.VAULT = self.tmp

    def _seat_entry(self, result, seat):
        for s in result["seats"]:
            if s["seat"] == seat:
                return s
        raise AssertionError(f"seat {seat} missing from relay_audit output")

    # --- 1. five-seat output shape -------------------------------------------

    def test_node_and_legacy_root_shape(self):
        self._bind_all_seats({})
        result = kicker.relay_audit()
        self.assertTrue(result["ok"])
        self.assertIn("generated_at", result)
        self.assertEqual(result["sync_dependency"], "syncthing")
        self.assertEqual(result["scope"]["total_run_count"], "all_run_directories")
        self.assertEqual(result["scope"]["status_counts"], "inspected_recent_runs_only")
        seats = result["seats"]
        # WORKER1-1: three node roots first, then the legacy seat roots that keep
        # historical run records readable under the forward-only rule.
        self.assertEqual(
            [s["seat"] for s in seats],
            ["delta", "charlie", "alpha", "worker1", "worker3", "localworker", "worker4", "worker2"],
        )
        for s in seats:
            for key in (
                "run_root", "run_root_exists", "from_seat_root", "from_seat_root_exists",
                "total_run_count", "inspected_run_count", "inspection_limit",
                "status_counts_scope", "status_counts", "stale_count",
                "most_recent_run", "most_recent_done_at",
            ):
                self.assertIn(key, s)

    # --- 2. empty/missing roots -----------------------------------------------

    def test_missing_root_reports_absent_without_error(self):
        self._bind_all_seats({})
        # empty_runs dir DOES exist but is empty; simulate a genuinely absent
        # root for one seat by pointing it at a path that was never created.
        kicker.ALPHA_RUNS = os.path.join(self.tmp, "does-not-exist", "runs")
        result = kicker.relay_audit()
        # WORKER1-1: the alpha node root is the one just pointed at a missing path
        worker2 = self._seat_entry(result, "alpha")
        self.assertFalse(worker2["run_root_exists"])
        self.assertEqual(worker2["total_run_count"], 0)
        self.assertEqual(worker2["status_counts"], {"done": 0, "running": 0, "launching": 0, "stale": 0})
        self.assertEqual(worker2["stale_count"], 0)
        self.assertIsNone(worker2["most_recent_run"])
        self.assertIsNone(worker2["most_recent_done_at"])

    def test_empty_existing_root(self):
        self._bind_all_seats({})
        result = kicker.relay_audit()
        worker1 = self._seat_entry(result, "worker1")
        self.assertTrue(worker1["run_root_exists"])
        self.assertEqual(worker1["total_run_count"], 0)
        self.assertIsNone(worker1["most_recent_run"])

    # --- 3. done/running/stale counts ------------------------------------------

    def test_done_running_stale_counts(self):
        os.environ["TOWER_STALE_AFTER_SECONDS"] = "60"
        root = self._seat_root("worker3")

        running_dir = os.path.join(root, "RUNNING")
        os.makedirs(running_dir)
        _status_json(running_dir)
        _touch(os.path.join(running_dir, "heartbeat"), age_seconds=5)

        stale_dir = os.path.join(root, "STALE")
        os.makedirs(stale_dir)
        _status_json(stale_dir)
        _touch(os.path.join(stale_dir, "heartbeat"), age_seconds=300)

        done_dir = os.path.join(root, "DONE")
        os.makedirs(done_dir)
        _status_json(done_dir)
        _touch(os.path.join(done_dir, "done"), content="0\n")

        self._bind_all_seats({"worker3": root})
        result = kicker.relay_audit()
        worker3 = self._seat_entry(result, "worker3")
        self.assertEqual(worker3["total_run_count"], 3)
        self.assertEqual(worker3["status_counts"]["running"], 1)
        self.assertEqual(worker3["status_counts"]["stale"], 1)
        self.assertEqual(worker3["status_counts"]["done"], 1)
        self.assertEqual(worker3["stale_count"], 1)
        self.assertIsNotNone(worker3["most_recent_done_at"])
        # Most recent run by mtime is whichever dir was touched last (DONE).
        self.assertEqual(worker3["most_recent_run"]["token"], "DONE")
        self.assertEqual(worker3["most_recent_run"]["status"], "done")

    def test_stale_reason_surfaced_on_most_recent_run(self):
        os.environ["TOWER_STALE_AFTER_SECONDS"] = "60"
        root = self._seat_root("worker1")
        stale_dir = os.path.join(root, "ONLYRUN")
        os.makedirs(stale_dir)
        _status_json(stale_dir)
        _touch(os.path.join(stale_dir, "heartbeat"), age_seconds=999)

        self._bind_all_seats({"worker1": root})
        result = kicker.relay_audit()
        worker1 = self._seat_entry(result, "worker1")
        self.assertEqual(worker1["most_recent_run"]["status"], "stale")
        self.assertEqual(worker1["most_recent_run"]["stale_reason"], "heartbeat_timeout")
        self.assertIn("age_seconds", worker1["most_recent_run"])
        self.assertIn("last_seen_at", worker1["most_recent_run"])

    # --- 4. bounded/clamped max_recent ------------------------------------------

    def test_max_recent_clamped_high_and_low(self):
        self._bind_all_seats({})
        self.assertEqual(kicker.relay_audit(max_recent=999)["max_recent"], 100)
        self.assertEqual(kicker.relay_audit(max_recent=0)["max_recent"], 1)
        self.assertEqual(kicker.relay_audit(max_recent=-5)["max_recent"], 1)
        self.assertEqual(kicker.relay_audit(max_recent="not-a-number")["max_recent"], 20)

    def test_max_recent_bounds_status_counts_but_not_total(self):
        root = self._seat_root("localworker")
        for i in range(5):
            d = os.path.join(root, f"TOK{i}")
            os.makedirs(d)
            _status_json(d)
            _touch(os.path.join(d, "heartbeat"), age_seconds=i)  # TOK0 newest

        self._bind_all_seats({"localworker": root})
        result = kicker.relay_audit(max_recent=2)
        localworker = self._seat_entry(result, "localworker")
        self.assertEqual(localworker["total_run_count"], 5)
        self.assertEqual(localworker["inspected_run_count"], 2)
        self.assertEqual(localworker["inspection_limit"], 2)
        self.assertEqual(localworker["status_counts_scope"], "inspected_recent_runs_only")
        # Only 2 most-recently-modified dirs get opened/classified.
        counted = sum(localworker["status_counts"].values())
        self.assertEqual(counted, localworker["inspected_run_count"])

    # --- 5. syncthing active/inactive/unknown -----------------------------------

    def test_syncthing_active(self):
        self._bind_all_seats({})
        fake = mock.Mock(stdout="active\n")
        with mock.patch("subprocess.run", return_value=fake) as m:
            result = kicker.relay_audit()
        self.assertEqual(result["syncthing_service"], "active")
        self.assertEqual(m.call_args.args[0], ["systemctl", "--user", "is-active", "syncthing.service"])

    def test_syncthing_inactive(self):
        self._bind_all_seats({})
        fake = mock.Mock(stdout="inactive\n")
        with mock.patch("subprocess.run", return_value=fake):
            result = kicker.relay_audit()
        self.assertEqual(result["syncthing_service"], "inactive")

    def test_syncthing_unknown_on_timeout(self):
        self._bind_all_seats({})
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=3)):
            result = kicker.relay_audit()
        self.assertEqual(result["syncthing_service"], "unknown")

    def test_syncthing_unknown_on_missing_binary(self):
        self._bind_all_seats({})
        with mock.patch("subprocess.run", side_effect=FileNotFoundError()):
            result = kicker.relay_audit()
        self.assertEqual(result["syncthing_service"], "unknown")

    # --- 6. no mutation of run directories --------------------------------------

    def test_no_mutation_of_run_directories(self):
        os.environ["TOWER_STALE_AFTER_SECONDS"] = "60"
        root = self._seat_root("worker4")
        run_dir = os.path.join(root, "TOK")
        os.makedirs(run_dir)
        _status_json(run_dir)
        _touch(os.path.join(run_dir, "heartbeat"), age_seconds=5)

        before = {}
        for name in ("status.json", "heartbeat"):
            p = os.path.join(run_dir, name)
            with open(p, encoding="utf-8") as f:
                before[name] = (f.read(), os.path.getmtime(p))

        self._bind_all_seats({"worker4": root})
        kicker.relay_audit()

        self.assertEqual(sorted(os.listdir(run_dir)), ["heartbeat", "status.json"])
        for name, (content, mtime) in before.items():
            p = os.path.join(run_dir, name)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), content)
            self.assertEqual(os.path.getmtime(p), mtime)

    # --- 7. vault_root / relay_root presence ------------------------------------

    def test_vault_and_relay_root_reported(self):
        self._bind_all_seats({})
        result = kicker.relay_audit()
        self.assertIn("vault_root", result)
        self.assertIn("vault_root_exists", result)
        self.assertIn("relay_root", result)
        self.assertTrue(result["relay_root_exists"])

    # --- 8. server import/compile -----------------------------------------------

    def test_server_module_imports_and_registers_relay_audit(self):
        import importlib
        import server as server_module
        importlib.reload(server_module)
        self.assertTrue(callable(server_module.relay_audit))
        tool = server_module.mcp._tool_manager.get_tool("relay_audit")
        self.assertIsNotNone(tool)


if __name__ == "__main__":
    unittest.main()

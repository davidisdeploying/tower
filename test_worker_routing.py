import json
import os
import tempfile
import subprocess
import sys
import unittest
from contextlib import nullcontext
from unittest import mock

import kicker
import server


class WorkerRoutingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.roots = {
            "delta": os.path.join(self.root, "from-delta", "runs"),
            "charlie": os.path.join(self.root, "from-charlie", "runs"),
            "alpha": os.path.join(self.root, "from-alpha", "runs"),
        }
        for root in self.roots.values():
            os.makedirs(root)
        self.patches = [
            mock.patch.object(kicker, "DELTA_RUNS", self.roots["delta"]),
            mock.patch.object(kicker, "CHARLIE_RUNS", self.roots["charlie"]),
            mock.patch.object(kicker, "ALPHA_RUNS", self.roots["alpha"]),
            mock.patch.object(
                server, "_WORKER_ROUTER_LOCK", os.path.join(self.root, "router.lock")
            ),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def active_run(self, seat, token, target_host="", target_scope="", lane="prompts"):
        run_dir = os.path.join(self.roots[seat], token)
        os.makedirs(run_dir)
        if target_host:
            with open(os.path.join(run_dir, "worker-routing.json"), "w") as handle:
                json.dump(
                    {
                        "seat": seat,
                        "target_host": target_host,
                        "target_scope": target_scope,
                        "lane": lane,
                    },
                    handle,
                )
        return {"source": seat, "token": token, "status": "running"}

    def test_target_local_idle_worker_wins(self):
        with mock.patch.object(kicker, "status", return_value={"ok": True, "runs": []}):
            result = server._select_worker(
                "delta", "repo:/home/david/loupe", "prompts"
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["seat"], "delta")
        self.assertFalse(result["cross_host"])

    def test_busy_local_worker_spills_to_idle_remote_worker(self):
        run = self.active_run("delta", "FLEET-WORKER1-BUILD-20260728-busy")
        with mock.patch.object(
            kicker, "status", return_value={"ok": True, "runs": [run]}
        ):
            result = server._select_worker(
                "delta", "repo:/home/david/loupe", "prompts"
            )
        self.assertTrue(result["ok"])
        # which idle node wins a tie is an arbitrary least-recently-used
        # ordering, so assert the property that matters: it spilled off the
        # busy target-local node and is therefore cross-host.
        self.assertNotEqual(result["seat"], "delta")
        self.assertIn(result["seat"], ("charlie", "alpha"))
        self.assertTrue(result["cross_host"])

    def test_different_repositories_on_one_host_can_run_concurrently(self):
        run = self.active_run("delta",
            "FLEET-WORKER1-BUILD-20260728-prospect",
            "delta",
            "repo:/home/david/prospect",
        )
        with mock.patch.object(
            kicker, "status", return_value={"ok": True, "runs": [run]}
        ):
            result = server._select_worker(
                "delta", "repo:/home/david/loupe", "prompts"
            )
        self.assertTrue(result["ok"])
        self.assertNotEqual(result["seat"], "delta")

    def test_same_repository_is_serialized_across_workers(self):
        run = self.active_run("delta",
            "FLEET-WORKER1-BUILD-20260728-loupe",
            "delta",
            "repo:/home/david/loupe",
        )
        with mock.patch.object(
            kicker, "status", return_value={"ok": True, "runs": [run]}
        ):
            result = server._select_worker(
                "delta", "repo:/home/david/loupe", "prompts"
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "target scope busy")

    def test_host_scope_serializes_all_mutation_on_target(self):
        run = self.active_run("delta",
            "FLEET-WORKER1-BUILD-20260728-host",
            "delta",
            "host:delta",
        )
        with mock.patch.object(
            kicker, "status", return_value={"ok": True, "runs": [run]}
        ):
            result = server._select_worker(
                "delta", "repo:/home/david/loupe", "prompts"
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "target scope busy")

    def test_auto_dispatch_injects_cross_host_context(self):
        route = {
            "ok": True,
            "seat": "charlie",
            "worker_host": "charlie",
            "target_host": "delta",
            "target_scope": "repo:/home/david/loupe",
            "cross_host": True,
            "reason": "least_recently_used_available_worker",
            "busy": {"worker1": [], "worker3": [], "worker2": []},
        }
        token = "FLEET-AUTO-BUILD-20260728-loupe"
        content = f"{token}\nwork only on Loupe\n{token}\n"
        with mock.patch.object(server, "_worker_router_lock", return_value=nullcontext()), \
             mock.patch.object(server, "_select_worker", return_value=route), \
             mock.patch.object(kicker, "kick_charlie", return_value={"ok": True}) as launch:
            result = server.dispatch(
                "auto",
                "prompts",
                token,
                content,
                target_host="delta",
                target_scope="repo:/home/david/loupe",
            )
        self.assertTrue(result["ok"])
        kwargs = launch.call_args.kwargs
        self.assertIn("ssh delta", kwargs["execution_context"])
        self.assertIn("does not expand", kwargs["execution_context"])
        self.assertEqual(kwargs["worker_routing"]["seat"], "charlie")
        self.assertEqual(result["worker_routing"]["target_host"], "delta")

    def test_auto_build_requires_target_and_scope(self):
        token = "FLEET-AUTO-BUILD-20260728-validation"
        content = f"{token}\nbody\n"
        self.assertEqual(
            server.dispatch("auto", "prompts", token, content)["error"],
            "target host required for auto seat",
        )
        self.assertEqual(
            server.dispatch(
                "auto", "prompts", token, content, target_host="delta"
            )["error"],
            "target scope required for auto build",
        )

    def test_planned_offline_target_and_workers_fail_closed(self):
        token = "FLEET-AUTO-RECON-20260728-offline"
        content = f"{token}\nbody\n{token}\n"
        with mock.patch.object(
            server, "_PLANNED_OFFLINE_NODES", frozenset({"charlie"})
        ), mock.patch.object(
            server, "_CLOUD_WORKER_SEATS", ("delta", "alpha")
        ):
            target = server.worker_routing_status(
                target_host="charlie", target_scope="host:charlie", lane="recon"
            )
            explicit = server.dispatch(
                "charlie", "recon", token, content, target_host="delta"
            )
            localworker = server.dispatch(
                "localworker", "recon", token, content, target_host="delta"
            )
            with mock.patch.object(
                kicker, "status", return_value={"ok": True, "runs": []}
            ):
                snapshot = server._worker_router_snapshot()

        self.assertEqual(target["error"], "target host planned offline")
        self.assertEqual(explicit["error"], "worker host planned offline")
        self.assertEqual(localworker["error"], "worker host planned offline")
        self.assertEqual(set(snapshot["busy"]), {"delta", "alpha"})

    def test_environment_excludes_workers_independently_of_targets(self):
        env = dict(os.environ, TOWER_PLANNED_OFFLINE_NODES="alpha",
                   TOWER_WORKER_EXCLUDED_NODES="charlie,alpha")
        code = "import json,server;print(json.dumps(list(server._CLOUD_WORKER_NODES)))"
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), ["delta"])
        env.pop("TOWER_WORKER_EXCLUDED_NODES")
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), ["delta", "charlie"])

    def test_worker_exclusion_preserves_charlie_target_and_delta_transport(self):
        token = "FLEET-RECON-20260929-worker-exclusion"
        content = f"{token}\nbody\n{token}\n"
        with mock.patch.object(server, "_PLANNED_OFFLINE_NODES", frozenset({"alpha"})), \
             mock.patch.object(server, "_WORKER_EXCLUDED_NODES", frozenset({"charlie", "alpha"})), \
             mock.patch.object(server, "_CLOUD_WORKER_SEATS", ("delta",)), \
             mock.patch.object(kicker, "status", return_value={"ok": True, "runs": []}), \
             mock.patch.object(kicker, "kick_charlie") as charlie_launch, \
             mock.patch.object(kicker, "kick_alpha") as alpha_launch, \
             mock.patch.object(kicker, "kick_localworker") as localworker_launch:
            route = server.worker_routing_status(target_host="charlie", lane="recon")
            self.assertTrue(route["ok"])
            self.assertEqual(route["worker_host"], "delta")
            self.assertEqual(route["target_host"], "charlie")
            self.assertTrue(route["cross_host"])
            self.assertEqual(server.dispatch("charlie", "recon", token, content,
                             target_host="delta")["error"], "worker host excluded")
            self.assertEqual(server.dispatch("worker3", "recon", token, content,
                             target_host="delta")["error"], "worker host excluded")
            self.assertEqual(server.dispatch("localworker", "recon", token, content,
                             target_host="charlie")["error"], "worker host excluded")
            self.assertEqual(server.dispatch("auto", "recon", token, content,
                             target_host="charlie", localworker_eligible=True)["error"],
                             "worker host excluded")
            self.assertEqual(server.dispatch("alpha", "recon", token, content,
                             target_host="delta")["error"], "worker host planned offline")
            self.assertEqual(server.worker_routing_status(target_host="alpha")["error"],
                             "target host planned offline")
            charlie_launch.assert_not_called()
            alpha_launch.assert_not_called()
            localworker_launch.assert_not_called()



if __name__ == "__main__":
    unittest.main()

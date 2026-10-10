"""TOWER-DISPATCH-1 regression gate.

Two concerns, one file, because they share a harness:

  A. The Localworker auto-route. `dispatch(seat="auto", target_host="charlie",
     provider="auto", localworker_eligible=True, ...)` raised
     `kick_localworker() got an unexpected keyword argument 'execution_context'`
     before launch, so the DL-9 probation could never record run #1. The
     fail-closed envelope validation that guards that route is CORRECT and is
     re-asserted here: a repair that opened the route by loosening validation
     would be a regression, not a fix.

  B. Multi-host collision scope. A run that mutates more than one host declares
     every (host, scope) pair via `additional_scopes` instead of naming only the
     most consequential host.
"""

import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from unittest import mock

import kicker
import server

ENVELOPE = (
    "FLEET_COMPACT_DELIVERY_V1_BEGIN\n"
    '{"tool_sequence": ["read_file", "finish_task_compact"], '
    '"required_response_markers": ["%s"]}\n'
    "FLEET_COMPACT_DELIVERY_V1_END"
)


class LocalworkerAutoRouteTestCase(unittest.TestCase):
    """Concern A — the auto route must launch, and stay fail-closed."""

    def compact_content(self, token):
        return f"{token}\n\n{ENVELOPE % token}\n\n{token}\n"

    def dispatch_localworker(self, token, content, **overrides):
        kwargs = dict(
            seat="auto",
            lane="recon",
            token=token,
            content=content,
            provider="auto",
            target_host="charlie",
            localworker_eligible=True,
        )
        kwargs.update(overrides)
        with mock.patch.object(
            server, "_worker_router_lock", return_value=nullcontext()
        ), mock.patch.object(
            kicker, "kick_localworker", return_value={"ok": True}
        ) as launch:
            result = server.dispatch(**kwargs)
        return result, launch

    def test_auto_route_launches_and_passes_routing_kwargs(self):
        """The exact defect: these kwargs used to raise TypeError before launch."""
        token = "FLEET-AUTO-RECON-20260803-localworker-auto"
        result, launch = self.dispatch_localworker(token, self.compact_content(token))
        self.assertTrue(result["ok"], result)
        launch.assert_called_once()
        kwargs = launch.call_args.kwargs
        self.assertIn("execution_context", kwargs)
        self.assertIn("worker_routing", kwargs)
        self.assertEqual(kwargs["worker_routing"]["seat"], "localworker")
        self.assertEqual(kwargs["worker_routing"]["worker_host"], "charlie")
        self.assertFalse(kwargs["worker_routing"]["cross_host"])
        self.assertEqual(
            kwargs["worker_routing"]["reason"], "localworker_first_compact_contract"
        )
        # Charlie-local work must never be told to self-SSH.
        self.assertIn("never self-SSH", kwargs["execution_context"])

    def test_localworker_is_not_given_cloud_provider_kwargs(self):
        token = "FLEET-AUTO-RECON-20260803-localworker-noprovider"
        _, launch = self.dispatch_localworker(token, self.compact_content(token))
        kwargs = launch.call_args.kwargs
        self.assertNotIn("provider", kwargs)
        self.assertNotIn("task_size", kwargs)

    def test_kick_localworker_forwards_routing_to_kick_remote(self):
        route = {"seat": "localworker", "target_host": "charlie"}
        with mock.patch.object(
            kicker, "_kick_remote", return_value={"ok": True}
        ) as remote:
            kicker.kick_localworker(
                "FLEET-LOCALWORKER-RECON-20260803-forward",
                "recon",
                "",
                execution_context="ROUTED CONTEXT",
                worker_routing=route,
            )
        kwargs = remote.call_args.kwargs
        self.assertEqual(kwargs["execution_context"], "ROUTED CONTEXT")
        self.assertEqual(kwargs["worker_routing"], route)
        self.assertEqual(kwargs["seat"], "localworker")

    # -- fail-closed validation must survive the repair --------------------

    def test_missing_envelope_still_fails_closed(self):
        token = "FLEET-AUTO-RECON-20260803-no-envelope"
        result, launch = self.dispatch_localworker(token, f"{token}\nbody\n{token}\n")
        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"], "localworker auto requires compact delivery contract"
        )
        launch.assert_not_called()

    def test_half_envelope_still_fails_closed(self):
        token = "FLEET-AUTO-RECON-20260803-half-envelope"
        content = f"{token}\nFLEET_COMPACT_DELIVERY_V1_BEGIN\n{{}}\n{token}\n"
        result, launch = self.dispatch_localworker(token, content)
        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"], "localworker auto requires compact delivery contract"
        )
        launch.assert_not_called()

    def test_localworker_eligibility_requires_auto_seat_provider_and_charlie(self):
        token = "FLEET-AUTO-RECON-20260803-guards"
        content = self.compact_content(token)
        for overrides, expected in (
            ({"seat": "localworker"}, "localworker eligibility requires auto seat"),
            ({"provider": "claude"}, "localworker eligibility requires auto provider"),
            ({"target_host": "delta"}, "localworker auto requires charlie target"),
        ):
            result, launch = self.dispatch_localworker(token, content, **overrides)
            self.assertFalse(result["ok"], overrides)
            self.assertEqual(result["error"], expected)
            launch.assert_not_called()


class MultiHostScopeTestCase(unittest.TestCase):
    """Concern B — every mutated host is declared and collision-checked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.roots = {
            seat: os.path.join(self.tmp.name, f"from-{seat}", "runs")
            for seat in ("delta", "charlie", "alpha")
        }
        for root in self.roots.values():
            os.makedirs(root)
        for patch in (
            mock.patch.object(kicker, "DELTA_RUNS", self.roots["delta"]),
            mock.patch.object(kicker, "CHARLIE_RUNS", self.roots["charlie"]),
            mock.patch.object(kicker, "ALPHA_RUNS", self.roots["alpha"]),
            mock.patch.object(
                server,
                "_WORKER_ROUTER_LOCK",
                os.path.join(self.tmp.name, "router.lock"),
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def active_run(self, seat, token, route):
        run_dir = os.path.join(self.roots[seat], token)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "worker-routing.json"), "w") as handle:
            json.dump({"seat": seat, "lane": "prompts", **route}, handle)
        return {"source": seat, "token": token, "status": "running"}

    def select(self, runs, **kwargs):
        with mock.patch.object(
            kicker, "status", return_value={"ok": True, "runs": runs}
        ):
            return server._select_worker(**kwargs)

    # -- parsing -----------------------------------------------------------

    def test_parse_accepts_host_qualified_claims(self):
        claims, error = server._parse_additional_scopes(
            "charlie:service:ollama.service, delta:repo:/home/david/loupe"
        )
        self.assertEqual(error, "")
        self.assertEqual(
            claims,
            [
                {"host": "charlie", "scope": "service:ollama.service"},
                {"host": "delta", "scope": "repo:/home/david/loupe"},
            ],
        )

    def test_parse_rejects_unknown_host_and_malformed_scope(self):
        _, error = server._parse_additional_scopes("nowhere:repo:/x")
        self.assertEqual(error, "bad additional scope host: nowhere")
        _, error = server._parse_additional_scopes("charlie:not-a-scope")
        self.assertEqual(error, "bad additional scope: charlie:not-a-scope")

    def test_merge_puts_primary_first_and_deduplicates(self):
        merged = server._merge_claims(
            "alpha",
            "repo:/home/david/tower",
            [
                {"host": "alpha", "scope": "repo:/home/david/tower"},
                {"host": "charlie", "scope": "service:ollama.service"},
            ],
        )
        self.assertEqual(
            merged,
            [
                {"host": "alpha", "scope": "repo:/home/david/tower"},
                {"host": "charlie", "scope": "service:ollama.service"},
            ],
        )

    # -- collision detection -----------------------------------------------

    def test_secondary_host_collision_is_detected(self):
        """The gap this closes: the collision is on the UNDECLARED second host."""
        run = self.active_run("charlie",
            "FLEET-AUTO-BUILD-20260803-ollama",
            {"target_host": "charlie", "target_scope": "service:ollama.service"},
        )
        result = self.select(
            [run],
            target_host="alpha",
            target_scope="repo:/home/david/tower",
            lane="prompts",
            claims=[
                {"host": "alpha", "scope": "repo:/home/david/tower"},
                {"host": "charlie", "scope": "service:ollama.service"},
            ],
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "target scope busy")

    def test_disjoint_secondary_scopes_still_run_concurrently(self):
        run = self.active_run("charlie",
            "FLEET-AUTO-BUILD-20260803-indexer",
            {"target_host": "charlie", "target_scope": "service:fleet-maint.service"},
        )
        result = self.select(
            [run],
            target_host="alpha",
            target_scope="repo:/home/david/tower",
            lane="prompts",
            claims=[
                {"host": "alpha", "scope": "repo:/home/david/tower"},
                {"host": "charlie", "scope": "service:ollama.service"},
            ],
        )
        self.assertTrue(result["ok"], result)

    def test_held_multi_host_claim_blocks_a_later_single_scope_run(self):
        """Collision is symmetric: the HELD run is the one with two claims."""
        run = self.active_run("alpha",
            "FLEET-AUTO-BUILD-20260803-multi",
            {
                "target_host": "alpha",
                "target_scope": "repo:/home/david/tower",
                "claims": [
                    {"host": "alpha", "scope": "repo:/home/david/tower"},
                    {"host": "charlie", "scope": "service:ollama.service"},
                ],
            },
        )
        result = self.select(
            [run],
            target_host="charlie",
            target_scope="service:ollama.service",
            lane="prompts",
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "target scope busy")

    def test_pre_claims_routes_are_still_honored(self):
        """Back-compat: a route written before this change has no claims key."""
        run = self.active_run("delta",
            "FLEET-WORKER1-BUILD-20260803-legacy",
            {"target_host": "delta", "target_scope": "repo:/home/david/loupe"},
        )
        result = self.select(
            [run],
            target_host="delta",
            target_scope="repo:/home/david/loupe",
            lane="prompts",
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "target scope busy")

    def test_recon_never_collides(self):
        run = self.active_run("charlie",
            "FLEET-AUTO-BUILD-20260803-busy",
            {"target_host": "charlie", "target_scope": "host:charlie"},
        )
        result = self.select(
            [run],
            target_host="charlie",
            target_scope="",
            lane="recon",
            claims=[{"host": "charlie", "scope": "service:ollama.service"}],
        )
        self.assertTrue(result["ok"], result)

    # -- dispatch plumbing --------------------------------------------------

    def test_dispatch_records_and_announces_every_claim(self):
        token = "FLEET-AUTO-BUILD-20260803-multihost"
        content = f"{token}\nbody\n{token}\n"
        route = {
            "ok": True,
            "seat": "worker2",
            "worker_host": "alpha",
            "target_host": "alpha",
            "target_scope": "repo:/home/david/tower",
            "cross_host": False,
            "reason": "target_local_worker_available",
            "busy": {"worker1": [], "worker3": [], "worker2": []},
        }
        with mock.patch.object(
            server, "_worker_router_lock", return_value=nullcontext()
        ), mock.patch.object(
            server, "_select_worker", return_value=route
        ) as select, mock.patch.object(
            kicker, "kick_alpha", return_value={"ok": True}
        ) as launch:
            result = server.dispatch(
                "auto",
                "prompts",
                token,
                content,
                target_host="alpha",
                target_scope="repo:/home/david/tower",
                additional_scopes="charlie:service:ollama.service",
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(
            select.call_args.args[3],
            [
                {"host": "alpha", "scope": "repo:/home/david/tower"},
                {"host": "charlie", "scope": "service:ollama.service"},
            ],
        )
        context = launch.call_args.kwargs["execution_context"]
        self.assertIn("repo:/home/david/tower", context)
        self.assertIn("service:ollama.service", context)
        self.assertIn("charlie", context)

    def test_dispatch_rejects_a_malformed_additional_scope(self):
        token = "FLEET-AUTO-BUILD-20260803-badscope"
        content = f"{token}\nbody\n{token}\n"
        with mock.patch.object(
            server, "_worker_router_lock", return_value=nullcontext()
        ):
            result = server.dispatch(
                "auto",
                "prompts",
                token,
                content,
                target_host="alpha",
                target_scope="repo:/home/david/tower",
                additional_scopes="charlie:ollama.service",
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "bad additional scope: charlie:ollama.service")


if __name__ == "__main__":
    unittest.main()

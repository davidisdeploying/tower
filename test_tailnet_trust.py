"""Trust-channel tests for the tailnet path
(FLEET-WORKER2-BUILD-20260811-tower-tailnet-path).

Tower used to have exactly one non-JWT way in: a genuine on-box loopback peer.
It now has two — loopback and a direct tailnet peer — so that fleet boxes and
David's own devices reach the vault over WireGuard instead of round-tripping
through Cloudflare's edge.

That is a deliberate widening of a security boundary, so these tests pin the
edges of it rather than the happy path:

  * the LAN is NOT the tailnet (192.168/16 and the 10.10.10/24 storage subnet
    must stay untrusted — only the WireGuard-authenticated CGNAT range counts);
  * the CGNAT range's own boundaries hold (100.63.255.255 out, 100.128.0.0 out);
  * an edge header defeats implicit trust from ANY peer, so a forged cf-ray
    cannot downgrade a request into the trusted branch and a request that really
    did come through the tunnel is always JWT-validated;
  * the default bind stays loopback-only, so the feature is opt-in from the unit
    and deleting the drop-in is a complete rollback.

Run with:
    python -m unittest test_tailnet_trust -v
"""
import os
import unittest
from unittest import mock

import anyio

import server


# Real fleet addresses — these are the peers the branch exists to admit.
CHARLIE_TAILNET = "100.64.0.35"
ALPHA_TAILNET = "100.64.0.36"
ALPHA_TAILNET_V6 = "fd7a:115c:a1e0::7637:d25"


class NormalizePeerTestCase(unittest.TestCase):
    def test_strips_v4_mapped_v6_prefix(self):
        """The latent gap this fixes: on a dual-stack listener loopback arrives as
        '::ffff:127.0.0.1', which the previous exact-string compare missed — the
        on-box caller would have been pushed onto the JWT path and rejected."""
        self.assertEqual(server.normalize_peer("::ffff:127.0.0.1"), "127.0.0.1")

    def test_strips_ipv6_zone_index(self):
        self.assertEqual(server.normalize_peer("fe80::1%eth0"), "fe80::1")

    def test_lowercases_and_trims(self):
        self.assertEqual(server.normalize_peer("  FD7A:115C:A1E0::1 "), "fd7a:115c:a1e0::1")

    def test_none_and_garbage_do_not_raise(self):
        self.assertEqual(server.normalize_peer(None), "")
        self.assertEqual(server.normalize_peer("not-an-ip"), "not-an-ip")


class LoopbackPeerTestCase(unittest.TestCase):
    def test_whole_127_block_is_loopback(self):
        for addr in ("127.0.0.1", "127.0.0.2", "127.1.2.3"):
            self.assertTrue(server.is_loopback_peer(addr), addr)

    def test_v6_loopback(self):
        self.assertTrue(server.is_loopback_peer("::1"))

    def test_v4_mapped_loopback(self):
        self.assertTrue(server.is_loopback_peer("::ffff:127.0.0.1"))

    def test_non_loopback_is_rejected(self):
        for addr in ("192.168.1.66", "8.8.8.8", CHARLIE_TAILNET, None, "garbage"):
            self.assertFalse(server.is_loopback_peer(addr), addr)


class TailnetPeerTestCase(unittest.TestCase):
    def test_fleet_tailnet_addresses_are_tailnet(self):
        for addr in (CHARLIE_TAILNET, ALPHA_TAILNET, "100.64.0.0", "100.64.0.255"):
            self.assertTrue(server.is_tailnet_peer(addr), addr)

    def test_tailnet_v6_ula_prefix(self):
        self.assertTrue(server.is_tailnet_peer(ALPHA_TAILNET_V6))

    def test_cgnat_boundaries_hold(self):
        """100.64.0.0/10 is 100.64.0.0 - 100.64.0.255. One address either side
        must fall outside, or the range has silently widened."""
        self.assertFalse(server.is_tailnet_peer("100.63.255.255"))
        self.assertFalse(server.is_tailnet_peer("100.128.0.0"))

    def test_lan_is_not_the_tailnet(self):
        """The whole point of following breadcrumbs rather than Panel: trust is
        narrowed to the WireGuard-authenticated range, NOT all of RFC1918. A LAN
        peer (guest wifi, a compromised IoT device) must never reach the vault."""
        for addr in ("192.168.1.66", "192.168.1.78", "192.0.2.1", "172.17.0.1"):
            self.assertFalse(server.is_tailnet_peer(addr), addr)

    def test_public_and_malformed_are_not_tailnet(self):
        for addr in ("8.8.8.8", "1.1.1.1", None, "", "garbage", "999.999.999.999"):
            self.assertFalse(server.is_tailnet_peer(addr), addr)

    def test_unrelated_v6_ula_is_not_this_tailnet(self):
        """fc00::/7 is the whole private-v6 space; only OUR tailnet's prefix counts."""
        self.assertFalse(server.is_tailnet_peer("fd00:dead:beef::1"))


class ClassifyPeerTrustTestCase(unittest.TestCase):
    def test_channels_are_named(self):
        self.assertEqual(server.classify_peer_trust("127.0.0.1", {}), "loopback")
        self.assertEqual(server.classify_peer_trust(CHARLIE_TAILNET, {}), "tailnet")

    def test_untrusted_peers_return_none(self):
        for addr in ("192.168.1.78", "8.8.8.8", None):
            self.assertIsNone(server.classify_peer_trust(addr, {}), addr)

    def test_edge_header_defeats_implicit_trust_from_any_peer(self):
        """The two-fact rule. cloudflared dials the origin from 127.0.0.1, so a
        tunnel-forwarded request looks loopback by peer alone; the edge headers are
        what distinguish it. This also means a forged cf-ray cannot downgrade a
        request INTO the trusted branch — it only ever forces stricter checking."""
        for peer in ("127.0.0.1", CHARLIE_TAILNET, ALPHA_TAILNET_V6):
            for header in ("cf-ray", "cf-connecting-ip"):
                self.assertIsNone(
                    server.classify_peer_trust(peer, {header: "x"}),
                    f"{peer} + {header}",
                )


class BindAddressesTestCase(unittest.TestCase):
    def test_default_is_loopback_only(self):
        """Unset env must reproduce the historical behavior exactly, so the tailnet
        path is opt-in from the unit and removing the drop-in is a full rollback."""
        env = dict(os.environ)
        env.pop("TOWER_BIND_ADDRESSES", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(server._bind_addresses(), ["127.0.0.1"])

    def test_csv_is_parsed_and_trimmed(self):
        with mock.patch.dict(
            os.environ,
            {"TOWER_BIND_ADDRESSES": " 127.0.0.1 , 100.64.0.36 "},
            clear=False,
        ):
            self.assertEqual(
                server._bind_addresses(), ["127.0.0.1", "100.64.0.36"]
            )

    def test_empty_value_falls_back_to_loopback(self):
        with mock.patch.dict(
            os.environ, {"TOWER_BIND_ADDRESSES": "  , ,"}, clear=False
        ):
            self.assertEqual(server._bind_addresses(), ["127.0.0.1"])

    def test_never_binds_wildcard_by_default(self):
        """0.0.0.0 would make the LAN reachable, and a LAN peer is not trusted by
        classify_peer_trust — but it would still be an unintended exposure."""
        self.assertNotIn("0.0.0.0", server._bind_addresses())


class AllowedHostsTestCase(unittest.TestCase):
    """The 421 trap: DNS-rebinding protection runs INSIDE the MCP transport, i.e.
    after the auth gate, and 421s any Host not on this list. A trusted tailnet peer
    that passes the gate and is then refused by the transport is a confusing
    failure, so the allowlist must cover every way a tailnet client addresses us."""

    def _allowed(self):
        return list(
            server.mcp.settings.transport_security.allowed_hosts
        )

    def test_tailnet_dns_name_is_allowed(self):
        self.assertIn(
            f"{server.TOWER_NODE}.{server.TAILNET_DNS_SUFFIX}:*", self._allowed()
        )

    def test_short_hostname_is_allowed(self):
        self.assertIn(f"{server.TOWER_NODE}:*", self._allowed())

    def test_every_bind_address_is_allowed(self):
        """Derived from the bind list precisely so the two cannot desync."""
        for addr in server.BIND_ADDRESSES:
            self.assertIn(f"{addr}:*", self._allowed())

    def test_public_hostnames_still_allowed(self):
        """The tunnel path must keep working — this change is additive."""
        for host in (
            "tower.example.com",
            "tower.example.com",
            "standby-tower.example.com",
        ):
            self.assertIn(host, self._allowed())


class MiddlewareIntegrationTestCase(unittest.TestCase):
    """End-to-end through the ASGI gate, following the idiom in
    test_operational_edge_cases.py."""

    def _run(self, peer, headers=()):
        called = []

        async def downstream(_scope, _receive, _send):
            called.append(True)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        events = []

        async def send(event):
            events.append(event)

        middleware = server.CfAccessJWTMiddleware.__new__(
            server.CfAccessJWTMiddleware
        )
        middleware.app = downstream
        middleware._jwks = None          # must never be consulted on a trusted path
        scope = {"type": "http", "client": (peer, 12345) if peer else None,
                 "headers": list(headers)}
        anyio.run(middleware, scope, receive, send)
        return bool(called), events

    def test_tailnet_peer_passes_without_a_jwt(self):
        passed, _ = self._run(CHARLIE_TAILNET)
        self.assertTrue(passed, "a direct tailnet peer should reach the app")

    def test_loopback_peer_still_passes(self):
        passed, _ = self._run("127.0.0.1")
        self.assertTrue(passed, "the pre-existing on-box path must not regress")

    def test_lan_peer_is_denied(self):
        passed, events = self._run("192.168.1.78")
        self.assertFalse(passed)
        self.assertEqual(events[0]["status"], 403)

    def test_public_peer_is_denied(self):
        passed, events = self._run("8.8.8.8")
        self.assertFalse(passed)
        self.assertEqual(events[0]["status"], 403)

    def test_tailnet_peer_with_forged_edge_header_is_not_implicitly_trusted(self):
        """It falls through to JWT validation and is denied for lack of an
        assertion — it does NOT get waved through on peer address alone."""
        passed, events = self._run(CHARLIE_TAILNET, [(b"cf-ray", b"forged")])
        self.assertFalse(passed)
        self.assertEqual(events[0]["status"], 403)

    def test_trusted_channel_is_logged_by_name(self):
        with self.assertLogs("tower.gate", level="INFO") as captured:
            self._run(CHARLIE_TAILNET)
        self.assertTrue(
            any("tailnet" in line for line in captured.output),
            "the channel must be visible in journalctl, not inferred",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

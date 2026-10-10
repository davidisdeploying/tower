import unittest
from unittest import mock

import anyio

import server


class TowerHealthStatusTestCase(unittest.TestCase):
    def test_ready_requires_vault_relay_and_index(self):
        with mock.patch.object(server.os.path, "isdir", return_value=True), mock.patch.object(
            server.os, "access", return_value=True
        ), mock.patch.object(server.vs, "index_metadata", return_value={"ok": True}):
            result = server.tower_health_status()

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["checks"],
            {
                "vaults_root": True,
                "relay_root": True,
                "semantic_index": True,
            },
        )

    def test_missing_dependency_fails_closed(self):
        def isdir(path):
            return path != server.kicker.VAULT

        with mock.patch.object(server.os.path, "isdir", side_effect=isdir), mock.patch.object(
            server.os, "access", return_value=True
        ), mock.patch.object(server.vs, "index_metadata", return_value={"ok": True}):
            result = server.tower_health_status()

        self.assertFalse(result["ok"])
        self.assertFalse(result["checks"]["relay_root"])

    def test_degraded_standby_allows_missing_semantic_index_only(self):
        with mock.patch.dict(
            server.os.environ, {"TOWER_STANDBY_DEGRADED": "1"}
        ), mock.patch.object(
            server.os.path, "isdir", return_value=True
        ), mock.patch.object(
            server.os, "access", return_value=True
        ), mock.patch.object(
            server.vs, "index_metadata", return_value={"ok": False}
        ):
            result = server.tower_health_status()

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "degraded-cold-standby")
        self.assertEqual(
            result["degraded_capabilities"],
            ["search_vault", "search_history", "index_metadata"],
        )

    def test_degraded_standby_still_requires_vault_and_relay(self):
        def isdir(path):
            return path != server.kicker.VAULT

        with mock.patch.dict(
            server.os.environ, {"TOWER_STANDBY_DEGRADED": "1"}
        ), mock.patch.object(
            server.os.path, "isdir", side_effect=isdir
        ), mock.patch.object(
            server.os, "access", return_value=True
        ), mock.patch.object(
            server.vs, "index_metadata", return_value={"ok": False}
        ):
            result = server.tower_health_status()

        self.assertFalse(result["ok"])

    def test_index_exception_fails_closed_without_exception_detail(self):
        with mock.patch.object(server.os.path, "isdir", return_value=True), mock.patch.object(
            server.os, "access", return_value=True
        ), mock.patch.object(
            server.vs, "index_metadata", side_effect=RuntimeError("secret path")
        ):
            result = server.tower_health_status()

        self.assertFalse(result["ok"])
        self.assertFalse(result["checks"]["semantic_index"])
        self.assertNotIn("secret path", str(result))

    def test_route_returns_200_or_503(self):
        async def call(status):
            with mock.patch.object(server, "tower_health_status", return_value=status):
                return await server.healthz(None)

        ready = anyio.run(
            call,
            {"ok": True, "service": "tower", "checks": {}},
        )
        failed = anyio.run(
            call,
            {"ok": False, "service": "tower", "checks": {}},
        )

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(failed.status_code, 503)


if __name__ == "__main__":
    unittest.main()

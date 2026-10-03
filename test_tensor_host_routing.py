import unittest
from unittest import mock

import kicker


class TensorHostRoutingTestCase(unittest.TestCase):
    def _launch(self, tower_node):
        spec = {
            "ok": True,
            "provider": "codex",
            "bin": None,
            "model": None,
            "reason": "test",
            "reservation_id": None,
        }
        with mock.patch.object(kicker, "TOWER_NODE", tower_node), mock.patch.object(
            kicker, "_entry_provider_spec", return_value=spec
        ), mock.patch.object(
            kicker, "_kick_remote", return_value={"ok": True}
        ) as launch, mock.patch.object(
            kicker, "_finish_routed_launch", return_value={"ok": True}
        ):
            kicker.kick_charlie(
                "FLEET-RECON-20260928-charlie-host-routing",
                lane="recon",
                prompt_content="bounded test",
            )
        return launch.call_args.kwargs

    def test_charlie_primary_uses_local_launch(self):
        kwargs = self._launch("charlie")
        self.assertTrue(kwargs["local"])
        self.assertIsNone(kwargs["host"])

    def test_other_tower_host_uses_charlie_ssh(self):
        kwargs = self._launch("alpha")
        self.assertFalse(kwargs["local"])
        self.assertEqual(kwargs["host"], "david@charlie")


if __name__ == "__main__":
    unittest.main()

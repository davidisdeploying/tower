import unittest
from unittest import mock

import kicker


class StandbyHostRoutingTestCase(unittest.TestCase):
    def _launch(self, tower_node):
        spec = {
            "ok": True,
            "provider": "claude",
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
            kicker.kick_alpha(
                "FLEET-WORKER2-RECON-20260728-standby-host-routing",
                lane="recon",
                prompt_content="bounded test",
            )
        return launch.call_args.kwargs

    def test_alpha_primary_uses_local_launch(self):
        kwargs = self._launch("alpha")
        self.assertTrue(kwargs["local"])
        self.assertIsNone(kwargs["host"])

    def test_lookout_standby_uses_alpha_ssh(self):
        kwargs = self._launch("bravo")
        self.assertFalse(kwargs["local"])
        self.assertEqual(kwargs["host"], "david@alpha")


if __name__ == "__main__":
    unittest.main()

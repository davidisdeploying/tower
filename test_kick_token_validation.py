import unittest
from unittest import mock

import kicker


class KickTokenValidationTestCase(unittest.TestCase):
    def test_all_seat_entrypoints_reject_before_io(self):
        entrypoints = (
            kicker.kick_delta,
            kicker.kick_charlie,
            kicker.kick_localworker,
            kicker.kick_worker4,
            kicker.kick_alpha,
        )
        token = "FLEET-WORKER2-BUILD-20260722-bad'token"

        with mock.patch("builtins.open") as opened, mock.patch.object(
            kicker.os, "makedirs"
        ) as makedirs:
            for entrypoint in entrypoints:
                with self.subTest(entrypoint=entrypoint.__name__):
                    result = entrypoint(token)
                    self.assertEqual(result, {"ok": False, "error": "malformed token"})

        opened.assert_not_called()
        makedirs.assert_not_called()

    def test_shell_and_path_metacharacters_are_rejected_before_io(self):
        malformed = (
            "FLEET-WORKER2-BUILD-20260722-quote'break",
            "FLEET-WORKER2-BUILD-20260722-$(touch-pwned)",
            "FLEET-WORKER2-BUILD-20260722-`id`",
            "FLEET-WORKER2-BUILD-20260722-../../escape",
            "FLEET-WORKER2-BUILD-20260722-slash/escape",
            "FLEET-WORKER2-BUILD-20260722-good\nBAD",
            "FLEET-WORKER2-BUILD-20260722-leading-",
            "FLEET-WORKER2-BUILD-20260722--leading",
            "FLEET-WORKER2-WORK-20260722-wrong-lane",
            "FLEET-WORKER2-BUILD-notadate-bad",
        )

        with mock.patch("builtins.open") as opened, mock.patch.object(
            kicker.os, "makedirs"
        ) as makedirs:
            for token in malformed:
                with self.subTest(token=repr(token)):
                    result = kicker.kick_alpha(token)
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["error"], "malformed token")

        opened.assert_not_called()
        makedirs.assert_not_called()

    def test_current_and_legacy_forms_pass_validation(self):
        valid = (
            "FLEET-WORKER2-BUILD-20260722-current-form",
            "FLEET-WORKER3-RECON-20260722-recon-form",
            "FLEET-BUILD-20260722-legacy-form",
            "FLEET-RECON-20260722-legacy-recon",
        )

        missing = FileNotFoundError("expected test sentinel")
        with mock.patch("builtins.open", side_effect=missing) as opened, mock.patch.object(
            kicker.os, "makedirs"
        ) as makedirs:
            for token in valid:
                with self.subTest(token=token):
                    result = kicker.kick_alpha(token)
                    self.assertFalse(result["ok"])
                    self.assertTrue(result["error"].startswith("cannot read staged prompt:"))

        self.assertEqual(opened.call_count, len(valid))
        makedirs.assert_not_called()

    def test_empty_token_keeps_specific_error(self):
        for token in ("", "   ", None):
            with self.subTest(token=token):
                self.assertEqual(
                    kicker.kick_alpha(token),
                    {"ok": False, "error": "empty token"},
                )


if __name__ == "__main__":
    unittest.main()

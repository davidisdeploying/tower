"""Focused tests for server._load_cf_aud and the CF-Access fail-closed gate
(FLEET-WORKER2-BUILD-20260710-cf-audience-config).

Audience values now come from the TOWER_CF_AUD env var (comma-separated),
never hardcoded in source. These tests use synthetic placeholder values only.
Run with:
    python -m unittest test_cf_audience_config -v
"""
import os
import unittest
from unittest import mock

import server


class LoadCfAudTestCase(unittest.TestCase):
    def test_single_audience(self):
        with mock.patch.dict(os.environ, {"TOWER_CF_AUD": "aud-one"}, clear=False):
            self.assertEqual(server._load_cf_aud(), ["aud-one"])

    def test_multiple_audiences_csv(self):
        with mock.patch.dict(os.environ, {"TOWER_CF_AUD": "aud-one,aud-two"}, clear=False):
            self.assertEqual(server._load_cf_aud(), ["aud-one", "aud-two"])

    def test_whitespace_stripped(self):
        with mock.patch.dict(os.environ, {"TOWER_CF_AUD": " aud-one , aud-two  "}, clear=False):
            self.assertEqual(server._load_cf_aud(), ["aud-one", "aud-two"])

    def test_empty_entries_ignored(self):
        with mock.patch.dict(os.environ, {"TOWER_CF_AUD": "aud-one,,aud-two,"}, clear=False):
            self.assertEqual(server._load_cf_aud(), ["aud-one", "aud-two"])

    def test_missing_env_var_yields_empty_list(self):
        env = dict(os.environ)
        env.pop("TOWER_CF_AUD", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(server._load_cf_aud(), [])

    def test_only_whitespace_yields_empty_list(self):
        with mock.patch.dict(os.environ, {"TOWER_CF_AUD": "   ,  , "}, clear=False):
            self.assertEqual(server._load_cf_aud(), [])

    def test_missing_env_var_logs_warning(self):
        env = dict(os.environ)
        env.pop("TOWER_CF_AUD", None)
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertLogs("tower.gate", level="WARNING") as cm:
                server._load_cf_aud()
            self.assertTrue(any("TOWER_CF_AUD" in line for line in cm.output))


class VerifyFailClosedTestCase(unittest.TestCase):
    """CfAccessJWTMiddleware._verify must fail closed when CF_AUD is empty,
    without ever restoring a source-code fallback audience."""

    def setUp(self):
        self._orig_cf_aud = server.CF_AUD

    def tearDown(self):
        server.CF_AUD = self._orig_cf_aud

    def _middleware(self):
        return server.CfAccessJWTMiddleware.__new__(server.CfAccessJWTMiddleware)

    def test_verify_raises_when_no_audience_configured(self):
        server.CF_AUD = []
        mw = self._middleware()
        with self.assertRaises(RuntimeError):
            mw._verify("irrelevant-token")

    def test_verify_does_not_call_jwks_when_no_audience_configured(self):
        server.CF_AUD = []
        mw = self._middleware()
        mw._jwks = mock.Mock()
        with self.assertRaises(RuntimeError):
            mw._verify("irrelevant-token")
        mw._jwks.get_signing_key_from_jwt.assert_not_called()

    def test_verify_proceeds_to_jwks_when_audience_configured(self):
        server.CF_AUD = ["aud-one", "aud-two"]
        mw = self._middleware()
        mw._jwks = mock.Mock()
        mw._jwks.get_signing_key_from_jwt.side_effect = RuntimeError("boom-from-jwks")
        with self.assertRaises(RuntimeError) as cm:
            mw._verify("irrelevant-token")
        self.assertEqual(str(cm.exception), "boom-from-jwks")
        mw._jwks.get_signing_key_from_jwt.assert_called_once_with("irrelevant-token")


if __name__ == "__main__":
    unittest.main()

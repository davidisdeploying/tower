"""Focused tests for the Localworker-only finish_task delivery footer
(FLEET-WORKER2-BUILD-20260720-localworker-delivery-footer).

Diagnosis: FLEET-WORKER3-RECON-20260720-gptoss-smoke-maxturns found a production
gpt-oss:20b Localworker run burning most of its tool-call budget manually
reconstructing response.md/responses.md/session-note writes that
fleet_agent.py's finish_task tool already automates, then never reaching
finish_task. This module proves:

  1. _build_localworker_footer now instructs finish_task-only delivery and
     excludes the manual response/log/session-note write instructions.
  2. The other four seats' footer functions (_build_footer, _build_worker3_footer,
     _build_worker4_footer, _build_worker2_footer) are byte-identical to golden output
     captured from kicker.py before this build touched it (backed by the
     localworker-delivery-footer backup dir's kicker.py.live for anyone re-deriving
     these constants).
  3. kick_localworker's generated run.sh still carries `--model gpt-oss:20b`
     (the approved model cutover, unrelated WIP this build must not disturb).
  4. server.summon_localworker/server.ask_localworker's docstrings (FLEET-WORKER2-
     BUILD-20260721-correct-localworker-session-note-contract) describe the same
     finish_task-only contract as _build_localworker_footer above, not the
     Option-A dual-voice session-note-editing contract the other four seats'
     docstrings carry.

Uses tempdirs + a monkeypatched subprocess.run (no live ssh/NAS/GPU). Run with:
    python -m unittest test_localworker_delivery_footer -v
"""
import os
import shutil
import subprocess
import tempfile
import unittest

import kicker
import server

TOKEN = "FLEET-TEST-TOKEN-golden"
LAUNCH_TOKEN = "FLEET-LOCALWORKER-BUILD-20260722-footer-fixture"
GOLDEN_ARGS = (TOKEN, "test-slug", "/tmp/resp.md", "/tmp/log.md", "/tmp/tokresp.md")
GOLDEN_ARGS_NO_NOTE = (TOKEN, "", "/tmp/resp.md", "/tmp/log.md", "/tmp/tokresp.md")

# Golden output captured from the pre-edit kicker.py (this build changes only
# _build_localworker_footer) with GOLDEN_ARGS / GOLDEN_ARGS_NO_NOTE.
_GOLDEN_WITH_NOTE = {
    "_build_footer": (
        "\n\n---\n\n## RELAY DELIVERY (appended by kick_delta — do this as the FINAL step)\n\n"
        "You were launched headless by the strategy seat for token `FLEET-TEST-TOKEN-golden`. "
        "When the work above is complete:\n\n"
        "1. OVERWRITE `/tmp/resp.md` with your concise response (what you did, key results, "
        "verification, anything open), beginning with a header line and ending by echoing the "
        "token `FLEET-TEST-TOKEN-golden`.\n"
        "2. ALSO write that same response to `/tmp/tokresp.md` (the per-token return slot — safe "
        "from being overwritten by a different run).\n"
        "3. APPEND that same response to `/tmp/log.md`, prefixed with a line of `---`.\n"
        "4. LOG to the session note `/home/david/Vaults/homelab-vault/sessions/test-slug.md` "
        "(READ-BEFORE-WRITE). Append ONLY your Worker1-owned sections — `## Execution log`, "
        "`## Blockers & fixes`, and `[Worker1]`-prefixed lines under `## Learnings` — in Option-A "
        "dual-voice. NEVER overwrite or rewrite the strategy-owned sections (`## Goal / context`, "
        "`## Decisions & rationale`, `## Open / next`). Append-only; if the note does not exist, "
        "create it with the standard headers and leave the strategy sections as empty stubs."
    ),
    "_build_worker3_footer": (
        "\n\n---\n\n## RELAY DELIVERY (appended by kick_charlie — do this as the FINAL step)\n\n"
        "You were launched headless on Worker3 (charlie) by the strategy seat for token "
        "`FLEET-TEST-TOKEN-golden`. When the work above is complete:\n\n"
        "1. OVERWRITE `/tmp/resp.md` with your concise response (what you did, key results, "
        "verification, anything open), beginning with a header line and ending by echoing the "
        "token `FLEET-TEST-TOKEN-golden`.\n"
        "2. ALSO write that same response to `/tmp/tokresp.md` (the per-token return slot — safe "
        "from being overwritten by a different run).\n"
        "3. APPEND that same response to `/tmp/log.md`, prefixed with a line of `---`.\n"
        "4. LOG to the session note `/home/david/Vaults/homelab-vault/sessions/test-slug.md` "
        "(READ-BEFORE-WRITE). Append ONLY your CC-owned sections — `## Execution log`, "
        "`## Blockers & fixes`, and `[CC]`-prefixed lines under `## Learnings` — in Option-A "
        "dual-voice. NEVER overwrite or rewrite the strategy-owned sections (`## Goal / context`, "
        "`## Decisions & rationale`, `## Open / next`). Append-only; if the note does not exist, "
        "create it with the standard headers and leave the strategy sections as empty stubs.\n\n"
        "Then echo the token `FLEET-TEST-TOKEN-golden` and HALT."
    ),
    "_build_worker4_footer": (
        "\n\n---\n\n## RELAY DELIVERY (appended by kick_worker4 — do this as the FINAL step)\n\n"
        "You were launched headless on Worker4 (MacBook) by the strategy seat for token "
        "`FLEET-TEST-TOKEN-golden`. When the work above is complete:\n\n"
        "1. OVERWRITE `/tmp/resp.md` with your concise response (what you did, key results, "
        "verification, anything open), beginning with a header line and ending by echoing the "
        "token `FLEET-TEST-TOKEN-golden`.\n"
        "2. ALSO write that same response to `/tmp/tokresp.md` (the per-token return slot — safe "
        "from being overwritten by a different run).\n"
        "3. APPEND that same response to `/tmp/log.md`, prefixed with a line of `---`.\n"
        "4. LOG to the session note `/home/david/Vaults/homelab-vault/sessions/test-slug.md` "
        "(READ-BEFORE-WRITE). Append ONLY your CC-owned sections — `## Execution log`, "
        "`## Blockers & fixes`, and `[CC]`-prefixed lines under `## Learnings` — in Option-A "
        "dual-voice. NEVER overwrite or rewrite the strategy-owned sections (`## Goal / context`, "
        "`## Decisions & rationale`, `## Open / next`). Append-only; if the note does not exist, "
        "create it with the standard headers and leave the strategy sections as empty stubs.\n\n"
        "Then echo the token `FLEET-TEST-TOKEN-golden` and HALT."
    ),
    "_build_worker2_footer": (
        "\n\n---\n\n## RELAY DELIVERY (appended by kick_alpha — do this as the FINAL step)\n\n"
        "You were launched headless on worker2 (local) by the strategy seat for token "
        "`FLEET-TEST-TOKEN-golden`. When the work above is complete:\n\n"
        "1. OVERWRITE `/tmp/resp.md` with your concise response (what you did, key results, "
        "verification, anything open), beginning with a header line and ending by echoing the "
        "token `FLEET-TEST-TOKEN-golden`.\n"
        "2. ALSO write that same response to `/tmp/tokresp.md` (the per-token return slot — safe "
        "from being overwritten by a different run).\n"
        "3. APPEND that same response to `/tmp/log.md`, prefixed with a line of `---`.\n"
        "4. LOG to the session note `/home/david/Vaults/homelab-vault/sessions/test-slug.md` "
        "(READ-BEFORE-WRITE). Append ONLY your Worker2-owned sections — `## Execution log`, "
        "`## Blockers & fixes`, and `[Worker2]`-prefixed lines under `## Learnings` — in Option-A "
        "dual-voice. NEVER overwrite or rewrite the strategy-owned sections (`## Goal / context`, "
        "`## Decisions & rationale`, `## Open / next`). Append-only; if the note does not exist, "
        "create it with the standard headers and leave the strategy sections as empty stubs.\n\n"
        "Then echo the token `FLEET-TEST-TOKEN-golden` and HALT."
    ),
}

# Same, but with no session_note supplied (step 4 dropped for the three seats
# that append "and HALT."; _build_footer has no HALT line even with a note).
_GOLDEN_NO_NOTE = {
    "_build_footer": (
        "\n\n---\n\n## RELAY DELIVERY (appended by kick_delta — do this as the FINAL step)\n\n"
        "You were launched headless by the strategy seat for token `FLEET-TEST-TOKEN-golden`. "
        "When the work above is complete:\n\n"
        "1. OVERWRITE `/tmp/resp.md` with your concise response (what you did, key results, "
        "verification, anything open), beginning with a header line and ending by echoing the "
        "token `FLEET-TEST-TOKEN-golden`.\n"
        "2. ALSO write that same response to `/tmp/tokresp.md` (the per-token return slot — safe "
        "from being overwritten by a different run).\n"
        "3. APPEND that same response to `/tmp/log.md`, prefixed with a line of `---`."
    ),
    "_build_worker3_footer": (
        "\n\n---\n\n## RELAY DELIVERY (appended by kick_charlie — do this as the FINAL step)\n\n"
        "You were launched headless on Worker3 (charlie) by the strategy seat for token "
        "`FLEET-TEST-TOKEN-golden`. When the work above is complete:\n\n"
        "1. OVERWRITE `/tmp/resp.md` with your concise response (what you did, key results, "
        "verification, anything open), beginning with a header line and ending by echoing the "
        "token `FLEET-TEST-TOKEN-golden`.\n"
        "2. ALSO write that same response to `/tmp/tokresp.md` (the per-token return slot — safe "
        "from being overwritten by a different run).\n"
        "3. APPEND that same response to `/tmp/log.md`, prefixed with a line of `---`.\n\n"
        "Then echo the token `FLEET-TEST-TOKEN-golden` and HALT."
    ),
    "_build_worker4_footer": (
        "\n\n---\n\n## RELAY DELIVERY (appended by kick_worker4 — do this as the FINAL step)\n\n"
        "You were launched headless on Worker4 (MacBook) by the strategy seat for token "
        "`FLEET-TEST-TOKEN-golden`. When the work above is complete:\n\n"
        "1. OVERWRITE `/tmp/resp.md` with your concise response (what you did, key results, "
        "verification, anything open), beginning with a header line and ending by echoing the "
        "token `FLEET-TEST-TOKEN-golden`.\n"
        "2. ALSO write that same response to `/tmp/tokresp.md` (the per-token return slot — safe "
        "from being overwritten by a different run).\n"
        "3. APPEND that same response to `/tmp/log.md`, prefixed with a line of `---`.\n\n"
        "Then echo the token `FLEET-TEST-TOKEN-golden` and HALT."
    ),
    "_build_worker2_footer": (
        "\n\n---\n\n## RELAY DELIVERY (appended by kick_alpha — do this as the FINAL step)\n\n"
        "You were launched headless on worker2 (local) by the strategy seat for token "
        "`FLEET-TEST-TOKEN-golden`. When the work above is complete:\n\n"
        "1. OVERWRITE `/tmp/resp.md` with your concise response (what you did, key results, "
        "verification, anything open), beginning with a header line and ending by echoing the "
        "token `FLEET-TEST-TOKEN-golden`.\n"
        "2. ALSO write that same response to `/tmp/tokresp.md` (the per-token return slot — safe "
        "from being overwritten by a different run).\n"
        "3. APPEND that same response to `/tmp/log.md`, prefixed with a line of `---`.\n\n"
        "Then echo the token `FLEET-TEST-TOKEN-golden` and HALT."
    ),
}


class LocalworkerFooterContentTestCase(unittest.TestCase):
    """_build_localworker_footer: finish_task-only, no manual delivery instructions."""

    def test_mentions_finish_task(self):
        out = kicker._build_localworker_footer(*GOLDEN_ARGS)
        self.assertIn("finish_task", out)
        self.assertIn(TOKEN, out)

    def test_excludes_manual_write_instructions(self):
        out = kicker._build_localworker_footer(*GOLDEN_ARGS)
        for banned in ("OVERWRITE", "APPEND", "ALSO write"):
            self.assertNotIn(banned, out, f"footer still tells Localworker to {banned!r}")
        # No reference to the response/log file paths — finish_task already
        # knows its own destinations from run.sh's --response-path/--responses-log.
        self.assertNotIn("/tmp/resp.md", out)
        self.assertNotIn("/tmp/log.md", out)
        self.assertNotIn("/tmp/tokresp.md", out)

    def test_explicitly_forbids_manual_delivery(self):
        out = kicker._build_localworker_footer(*GOLDEN_ARGS)
        self.assertIn("Do NOT manually write", out)
        self.assertIn("response.md", out)
        self.assertIn("latest_response.md", out)
        self.assertIn("responses.md", out)

    def test_session_note_supplied_forbids_manual_edit_not_option_a(self):
        out = kicker._build_localworker_footer(TOKEN, "test-slug", "/tmp/resp.md", "/tmp/log.md", "/tmp/tokresp.md")
        # Session note path is surfaced, but not with the other seats'
        # append-only Option-A editing instructions (no supported mechanism exists).
        self.assertIn("session note", out.lower())
        self.assertNotIn("READ-BEFORE-WRITE", out)
        self.assertNotIn("Option-A", out)
        self.assertNotIn("[Localworker]", out)

    def test_no_session_note_omits_session_note_step(self):
        out = kicker._build_localworker_footer(TOKEN, "", "/tmp/resp.md", "/tmp/log.md", "/tmp/tokresp.md")
        self.assertNotIn("session note", out.lower())

    def test_no_halt_instruction(self):
        # The generic footers end with "echo token and HALT" (a Claude Code
        # convention with no meaning to fleet_agent.py's tool loop, which
        # simply stops on a finish_task call). Localworker's should not carry it.
        out = kicker._build_localworker_footer(*GOLDEN_ARGS)
        self.assertNotIn("HALT", out)


class OtherSeatFootersUnchangedTestCase(unittest.TestCase):
    """Byte-identical diff of the four non-Localworker footer builders vs. golden
    output captured from kicker.py before this build touched it."""

    def test_footers_byte_identical_to_golden(self):
        for fn_name, expected in _GOLDEN_WITH_NOTE.items():
            live_out = getattr(kicker, fn_name)(*GOLDEN_ARGS)
            self.assertEqual(live_out, expected, f"{fn_name} output changed vs. golden")

    def test_footers_byte_identical_no_session_note(self):
        for fn_name, expected in _GOLDEN_NO_NOTE.items():
            live_out = getattr(kicker, fn_name)(*GOLDEN_ARGS_NO_NOTE)
            self.assertEqual(live_out, expected, f"{fn_name} output changed vs. golden (no session note)")


class KickLocalworkerRunShTestCase(unittest.TestCase):
    """kick_localworker's generated run.sh/prompt.txt, with ssh mocked out —
    no live launch, no NAS, no GPU."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fleet-localworker-footer-test-")
        self.vault = os.path.join(self.tmp, "vault")
        os.makedirs(self.vault)

        self._orig = {
            "VAULT": kicker.VAULT,
            "LOCALWORKER_RUNS": kicker.LOCALWORKER_RUNS,
            "LOCALWORKER_TRANSCRIPTS": kicker.LOCALWORKER_TRANSCRIPTS,
            "subprocess_run": kicker.subprocess.run,
        }
        kicker.VAULT = self.vault
        kicker.LOCALWORKER_RUNS = os.path.join(self.vault, "from-localworker", "runs")
        kicker.LOCALWORKER_TRANSCRIPTS = os.path.join(self.vault, "from-localworker", "transcripts")

        staged_dir = os.path.join(self.vault, "to-localworker", "prompts")
        os.makedirs(staged_dir)
        with open(os.path.join(staged_dir, "latest.md"), "w", encoding="utf-8") as f:
            f.write(f"# Build\n\n{LAUNCH_TOKEN}\n")

        def _fake_run(cmd, **kwargs):
            # Simulate the remote ssh launch always reporting the success marker
            # without touching any real host.
            return subprocess.CompletedProcess(cmd, 0, stdout="LOCALWORKER_LAUNCHED\n", stderr="")

        kicker.subprocess.run = _fake_run

    def tearDown(self):
        kicker.VAULT = self._orig["VAULT"]
        kicker.LOCALWORKER_RUNS = self._orig["LOCALWORKER_RUNS"]
        kicker.LOCALWORKER_TRANSCRIPTS = self._orig["LOCALWORKER_TRANSCRIPTS"]
        kicker.subprocess.run = self._orig["subprocess_run"]
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_sh_still_has_gpt_oss_model_flag(self):
        result = kicker.kick_localworker(LAUNCH_TOKEN, "prompts", "")
        self.assertTrue(result.get("ok"), result)
        run_sh_path = os.path.join(kicker.LOCALWORKER_RUNS, LAUNCH_TOKEN, "run.sh")
        with open(run_sh_path, "r", encoding="utf-8") as f:
            run_sh = f.read()
        self.assertIn("--model gpt-oss:20b", run_sh)
        self.assertIn("fleet_agent.py", run_sh)

    def test_prompt_txt_carries_finish_task_footer_not_manual(self):
        result = kicker.kick_localworker(LAUNCH_TOKEN, "prompts", "")
        self.assertTrue(result.get("ok"), result)
        prompt_path = os.path.join(kicker.LOCALWORKER_RUNS, LAUNCH_TOKEN, "prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        self.assertIn("finish_task", prompt)
        self.assertNotIn("OVERWRITE", prompt)

    def test_compact_prompt_gets_compact_footer_without_legacy_finish_instruction(self):
        compact = "\n".join(
            (
                LAUNCH_TOKEN,
                "FLEET_COMPACT_DELIVERY_V1_BEGIN",
                '{"tool_sequence":["read_file","finish_task_compact"],'
                '"required_response_markers":["## VERDICT","## Evidence",'
                f'"## Rollback","{LAUNCH_TOKEN}"]}}',
                "FLEET_COMPACT_DELIVERY_V1_END",
                LAUNCH_TOKEN,
            )
        )
        result = kicker.kick_localworker(
            LAUNCH_TOKEN, "prompts", "", prompt_content=compact
        )
        self.assertTrue(result.get("ok"), result)
        prompt_path = os.path.join(
            kicker.LOCALWORKER_RUNS, LAUNCH_TOKEN, "prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        self.assertIn("finish_task_compact", prompt)
        self.assertIn("Do NOT call the legacy `finish_task` tool", prompt)
        self.assertNotIn("Call the `finish_task` tool exactly once", prompt)


class LocalworkerToolDocstringContractTestCase(unittest.TestCase):
    """Localworker's delivery footer must describe the real finish_task-only
    contract, not the cloud workers' Option-A dual-voice session-note-logging
    contract (FLEET-WORKER2-BUILD-20260721-correct-localworker-session-note-contract).

    WORKER1-1 removed the per-seat summon_/ask_ tools, so this now asserts against the
    footer builder that actually carries the contract into the worker prompt."""

    def test_no_false_option_a_logging_claim(self):
        footer = kicker._build_localworker_footer("FLEET-TEST-RECON-20260818-x", "recon", "", "")
        self.assertNotIn("told to log its owned sections there", footer)
        self.assertNotIn("Option-A", footer)

    def test_states_leave_alone_finish_task_contract(self):
        footer = kicker._build_localworker_footer("FLEET-TEST-RECON-20260818-x", "recon", "", "")
        self.assertIn("finish_task", footer)


if __name__ == "__main__":
    unittest.main()

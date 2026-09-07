"""Regression test: reaching the tool-round limit must not leave orphan tool_calls.

Root cause (DeepSeek-compatible OpenAI endpoints): when the model returns
tool_calls in a round where ``max_tool_rounds`` is already exhausted, Moonshine
appended the assistant tool-call message to the provider transcript and then
``break`` before executing the tools / appending the matching tool messages.
The next provider request then contained an assistant message with ``tool_calls``
that had no following tool messages, which strict OpenAI-compatible servers
reject with HTTP 400 ("insufficient tool messages following tool_calls").
"""

from __future__ import annotations

import sys
import tempfile
import unittest

from moonshine.app import MoonshineApp
from moonshine.providers import ProviderResponse, ProviderStreamEvent, ProviderToolCall


def _make_tempdir():
    """TemporaryDirectory(ignore_cleanup_errors=...) requires Python 3.10+; keep 3.8 compat."""
    kwargs = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
    return tempfile.TemporaryDirectory(**kwargs)


class ScriptedProvider(object):
    """Minimal scripted provider used to drive the agent loop without a live model."""

    def __init__(self, scripted_responses):
        self.scripted_responses = list(scripted_responses)
        self.calls = []

    def stream_generate(self, *, system_prompt, messages, tool_schemas=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": [dict(item) for item in messages],
                "tool_schemas": list(tool_schemas or []),
            }
        )
        if not self.scripted_responses:
            raise AssertionError("ScriptedProvider ran out of scripted responses")
        step = self.scripted_responses.pop(0)
        for chunk in step.get("chunks", []):
            yield ProviderStreamEvent(type="text_delta", text=chunk)
        yield ProviderStreamEvent(type="response", response=step.get("response", ProviderResponse()))


def _orphan_tool_call_groups(messages):
    """Return (assistant_index, tool_call_id) groups that lack a following tool message."""
    orphans = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        following_tool_ids = set()
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            following_tool_ids.add(messages[j].get("tool_call_id"))
            j += 1
        for tool_call in message["tool_calls"]:
            call_id = tool_call.get("id")
            if call_id not in following_tool_ids:
                orphans.append((i, call_id))
    return orphans


class ProviderMessageSanitizerTest(unittest.TestCase):
    """Unit tests for the send-time sanitizer that guarantees protocol validity."""

    def setUp(self):
        self.temp_dir = _make_tempdir()
        self.addCleanup(self.temp_dir.cleanup)
        self.app = MoonshineApp(home=self.temp_dir.name)
        self.state = self.app.start_shell_state(mode="research", project_slug="anderson_conjecture")

    def _sanitize(self, messages):
        return self.app.agent._sanitize_provider_messages(messages)

    def test_well_formed_tool_pairing_is_unchanged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "let me check", "tool_calls": [{"id": "call-a", "function": {}}]},
            {"role": "tool", "tool_call_id": "call-a", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        cleaned = self._sanitize(messages)
        self.assertEqual(cleaned, messages)

    def test_orphan_tool_calls_are_demoted_to_plain_text(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "trying a tool", "tool_calls": [{"id": "call-1", "function": {}}]},
            {"role": "user", "content": "next turn, no tool result ever arrived"},
        ]
        cleaned = self._sanitize(messages)
        assistant = cleaned[1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertNotIn("tool_calls", assistant)
        self.assertIn("trying a tool", assistant["content"])

    def test_orphan_tool_results_are_folded_into_the_demoted_message(self):
        messages = [
            {"role": "assistant", "content": "checking", "tool_calls": [
                {"id": "call-a", "function": {}}, {"id": "call-b", "function": {}}]},
            {"role": "tool", "tool_call_id": "call-a", "content": "result a"},
            {"role": "user", "content": "continue"},
        ]
        cleaned = self._sanitize(messages)
        self.assertEqual(len(cleaned), 2)  # demoted assistant + trailing user; orphan tool folded away
        self.assertEqual(cleaned[0]["role"], "assistant")
        self.assertNotIn("tool_calls", cleaned[0])
        self.assertIn("result a", cleaned[0]["content"])
        self.assertIn("call-a", cleaned[0]["content"])  # the folded id must survive for traceability
        self.assertEqual(cleaned[1]["role"], "user")

    def test_orphan_with_no_content_gets_placeholder_text(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-x", "function": {}}]},
            {"role": "user", "content": "go on"},
        ]
        cleaned = self._sanitize(messages)
        self.assertEqual(cleaned[0]["role"], "assistant")
        self.assertNotIn("tool_calls", cleaned[0])
        # empty-content assistant messages are rejected by some strict endpoints,
        # so the placeholder must be a concrete non-empty marker
        self.assertEqual(cleaned[0]["content"], "[tool calls not executed]")

    def test_empty_input_returns_empty(self):
        self.assertEqual(self._sanitize([]), [])

    def test_lone_tool_message_without_preceding_tool_calls_is_demoted(self):
        """A role=tool message answering no assistant tool_calls is rejected by strict
        endpoints too, so it must be demoted to plain assistant text."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call-x", "content": "orphan result"},
            {"role": "user", "content": "go on"},
        ]
        cleaned = self._sanitize(messages)
        self.assertFalse(any(m.get("role") == "tool" for m in cleaned))
        self.assertEqual(cleaned[0]["role"], "user")
        self.assertEqual(cleaned[1]["role"], "assistant")
        self.assertNotIn("tool_calls", cleaned[1])
        self.assertIn("orphan result", cleaned[1]["content"])
        self.assertEqual(cleaned[2]["role"], "user")

    def test_stray_tool_messages_after_a_paired_group_are_folded(self):
        """All declared calls are answered, but an extra tool message with an unknown
        id trails the group: keep the pair intact, demote only the stray result."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-a", "function": {}}]},
            {"role": "tool", "tool_call_id": "call-a", "content": "ok"},
            {"role": "tool", "tool_call_id": "call-EXTRA", "content": "stray"},
            {"role": "user", "content": "next"},
        ]
        cleaned = self._sanitize(messages)
        self.assertEqual(cleaned[0].get("tool_calls"), [{"id": "call-a", "function": {}}])
        self.assertEqual(cleaned[1], {"role": "tool", "tool_call_id": "call-a", "content": "ok"})
        self.assertEqual(cleaned[2]["role"], "assistant")
        self.assertNotIn("tool_calls", cleaned[2])
        self.assertIn("stray", cleaned[2]["content"])
        self.assertEqual(cleaned[3]["role"], "user")
        self.assertEqual(_orphan_tool_call_groups(cleaned), [])

    def test_empty_tool_calls_list_is_dropped(self):
        messages = [{"role": "assistant", "content": "x", "tool_calls": []}]
        cleaned = self._sanitize(messages)
        self.assertNotIn("tool_calls", cleaned[0])
        self.assertEqual(cleaned[0]["content"], "x")

    def test_sanitize_is_idempotent(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "trying", "tool_calls": [{"id": "call-1", "function": {}}]},
            {"role": "tool", "tool_call_id": "call-9", "content": "stray"},
            {"role": "user", "content": "next"},
        ]
        once = self._sanitize(messages)
        twice = self._sanitize(once)
        self.assertEqual(once, twice)


class ToolRoundLimitOrphanRegressionTest(unittest.TestCase):
    """Ensure tool-round-limit finalization never produces orphan tool_calls."""

    def setUp(self):
        self.temp_dir = _make_tempdir()
        self.addCleanup(self.temp_dir.cleanup)
        self.app = MoonshineApp(home=self.temp_dir.name)
        self.state = self.app.start_shell_state(mode="research", project_slug="anderson_conjecture")

    def _assert_no_orphan_tool_calls_across_provider_calls(self, provider):
        """Assert every provider call's messages keep assistant tool_calls paired."""
        self.assertTrue(provider.calls, "provider was never called")
        for call_index, call in enumerate(provider.calls):
            orphans = _orphan_tool_call_groups(call["messages"])
            self.assertEqual(
                orphans,
                [],
                "provider call %s sent orphan tool_calls (DeepSeek returns HTTP 400 for these): %s"
                % (call_index, orphans),
            )

    def test_tool_round_limit_keeps_validation_feedback_path(self):
        """Budget exhaustion must not skip the invalid-batch / no-executable feedback path.

        These are validation-feedback paths: they append synthetic tool results and do
        NOT increment tool_rounds, so they must still run once the tool-round budget is
        exhausted. Before the fix, the max_tool_rounds guard `break` before reaching
        them, denying the model a chance to repair an invalid batch.

        Note the budget stays a HARD cap on real executions: once exhausted, a
        subsequently repaired valid call does NOT execute — the guard finalizes the
        run instead. Letting repaired calls run past the cap would make the budget
        bypassable by alternating invalid/valid batches (validation retries reset on
        every valid batch), so only the feedback path survives exhaustion.
        """
        self.app.config.agent.max_tool_rounds = 1  # exhaust after the first tool execution
        self.app.config.agent.max_model_rounds = 6

        provider = ScriptedProvider(
            [
                # round 1: a valid tool call -> executes (tool_rounds 0 -> 1, cap reached).
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="query_memory",
                                arguments={"query": "Krull dimension first", "project_slug": "anderson_conjecture"},
                                call_id="call-1",
                            )
                        ]
                    )
                },
                # round 2: budget exhausted, but the model returns an INVALID tool batch.
                # The guard must NOT preempt the validation-feedback path: the model should
                # still get the synthetic "unknown tool" result so it can repair.
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="totally_unknown_tool",
                                arguments={"query": "local methods", "project_slug": "anderson_conjecture"},
                                call_id="call-2",
                            )
                        ]
                    )
                },
                # round 3: the model repairs and returns a valid tool call. The budget is
                # a hard cap, so the guard gates this round and finalizes instead of
                # executing it — only the feedback path (round 2) survives exhaustion.
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="query_memory",
                                arguments={"query": "local methods", "project_slug": "anderson_conjecture"},
                                call_id="call-3",
                            )
                        ]
                    )
                },
                # finalization pass: plain text.
                {
                    "chunks": ["Finalized the run."],
                    "response": ProviderResponse(content="Finalized the run."),
                },
            ]
        )
        self.app.agent.provider = provider

        events = list(self.app.ask_stream("Run tools until the budget hits.", self.state))
        status_texts = [event.text for event in events if event.type == "status"]

        # The invalid batch must still surface its validation feedback (synthetic result),
        # i.e. the guard must not preempt the path that tells the model the tool is unknown.
        tool_errors = [e for e in events if e.type == "tool_error"]
        self.assertTrue(
            tool_errors,
            "validation feedback for the invalid batch was skipped; expected an "
            "'unknown tool' tool_result error to reach the model",
        )
        self.assertTrue(
            any("unknown tool" in str(e.payload.get("error", "")).lower() for e in tool_errors),
            "expected the error to mention an unknown tool",
        )
        # The round-1 call really executed; the repaired call-3 was gated by the hard
        # budget cap, so exactly one tool ran and no tool_call event exists for call-3.
        executed_call_ids = [e.payload.get("call_id") for e in events if e.type == "tool_call"]
        self.assertEqual(executed_call_ids, ["call-1"])
        tool_results = [e for e in events if e.type == "tool_result"]
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0].payload.get("call_id"), "call-1")
        self.assertTrue(any("tool round limit" in event.text.lower() for event in events if event.type == "status"))
        self.assertEqual(events[-1].text, "Finalized the run.")
        self._assert_no_orphan_tool_calls_across_provider_calls(provider)

    def test_invalid_batch_repair_executes_within_budget(self):
        """Normal repair flow: while budget remains, an invalid batch gets synthetic
        feedback (no tool_round consumed) and the repaired valid call then executes."""
        self.app.config.agent.max_tool_rounds = 1
        self.app.config.agent.max_model_rounds = 6

        provider = ScriptedProvider(
            [
                # round 1: invalid batch -> feedback, tool_rounds stays 0.
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="totally_unknown_tool",
                                arguments={"query": "x"},
                                call_id="bad-1",
                            )
                        ]
                    )
                },
                # round 2: repaired valid call -> budget remains, so it executes.
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="query_memory",
                                arguments={"query": "repaired", "project_slug": "anderson_conjecture"},
                                call_id="good-1",
                            )
                        ]
                    )
                },
                {
                    "response": ProviderResponse(content="Repair flow done."),
                },
            ]
        )
        self.app.agent.provider = provider

        events = list(self.app.ask_stream("Repair the batch.", self.state))

        self.assertTrue(any(e.type == "tool_error" for e in events))
        executed_call_ids = [e.payload.get("call_id") for e in events if e.type == "tool_call"]
        self.assertEqual(executed_call_ids, ["good-1"])
        self.assertEqual(events[-1].text, "Repair flow done.")
        self._assert_no_orphan_tool_calls_across_provider_calls(provider)

    def test_tool_round_limit_preserves_capped_round_text_for_finalization(self):
        """A capped response's text must reach the finalization request context.

        When the model returns text AND tool_calls after the budget is exhausted,
        the guard must skip the tool_calls (no orphan) but still keep the text as
        a plain assistant message in the provider conversation — otherwise the
        finalization pass cannot see what the model already wrote in that round.
        """
        self.app.config.agent.max_tool_rounds = 1  # exhaust after the first tool execution
        self.app.config.agent.max_model_rounds = 6

        provider = ScriptedProvider(
            [
                # round 1: a valid tool call -> executes (tool_rounds 0 -> 1, cap reached).
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="query_memory",
                                arguments={"query": "first", "project_slug": "anderson_conjecture"},
                                call_id="call-1",
                            )
                        ]
                    )
                },
                # round 2: text + tool call with the cap reached. The tool call is
                # gated (never appended, never executed); the text must be kept.
                {
                    "response": ProviderResponse(
                        content="CAPPED_ROUND_SENTINEL: local methods suffice.",
                        tool_calls=[
                            ProviderToolCall(
                                name="query_memory",
                                arguments={"query": "second", "project_slug": "anderson_conjecture"},
                                call_id="call-2",
                            )
                        ],
                    )
                },
                # finalization pass: plain text.
                {
                    "chunks": ["Finalized the run."],
                    "response": ProviderResponse(content="Finalized the run."),
                },
            ]
        )
        self.app.agent.provider = provider

        events = list(self.app.ask_stream("Run tools until capped with text.", self.state))

        self.assertTrue(any("tool round limit" in event.text.lower() for event in events if event.type == "status"))
        # The capped round's text is visible in the finalization request context.
        finalization_messages = provider.calls[-1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "assistant"
                and not message.get("tool_calls")
                and "CAPPED_ROUND_SENTINEL" in str(message.get("content") or "")
                for message in finalization_messages
            ),
            "finalization request did not include the capped round's assistant text",
        )
        self._assert_no_orphan_tool_calls_across_provider_calls(provider)

    def test_tool_round_limit_does_not_leave_orphan_tool_calls(self):
        self.app.config.agent.max_tool_rounds = 1  # exhaust after the first tool execution
        self.app.config.agent.max_model_rounds = 6

        provider = ScriptedProvider(
            [
                # round 1: a valid tool call -> executes (tool_rounds 0 -> 1, cap reached).
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="query_memory",
                                arguments={"query": "Krull dimension first", "project_slug": "anderson_conjecture"},
                                call_id="call-1",
                            )
                        ]
                    )
                },
                # round 2: the model wants to call a tool again, but the tool-round cap is hit.
                # Before the fix, the assistant tool-call message was appended and then dropped
                # without a following tool message -> orphan -> HTTP 400 on the next request.
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="query_memory",
                                arguments={"query": "local methods", "project_slug": "anderson_conjecture"},
                                call_id="call-2",
                            )
                        ]
                    )
                },
                # finalization pass: plain text, no tools.
                {
                    "chunks": ["Finalized the run."],
                    "response": ProviderResponse(content="Finalized the run."),
                },
            ]
        )
        self.app.agent.provider = provider

        events = list(self.app.ask_stream("Run the tool until the round limit.", self.state))

        self.assertTrue(any("tool round limit" in event.text.lower() for event in events if event.type == "status"))
        self._assert_no_orphan_tool_calls_across_provider_calls(provider)
        self.assertEqual(events[-1].text, "Finalized the run.")

    def test_multiple_parallel_tool_calls_stay_paired(self):
        """Sanity: parallel tool_calls in one response all get matching tool messages."""
        self.app.config.agent.max_tool_rounds = 2
        self.app.config.agent.max_model_rounds = 6

        provider = ScriptedProvider(
            [
                {
                    "response": ProviderResponse(
                        tool_calls=[
                            ProviderToolCall(
                                name="query_memory",
                                arguments={"query": "a", "project_slug": "anderson_conjecture"},
                                call_id="call-a",
                            ),
                            ProviderToolCall(
                                name="read_runtime_file",
                                arguments={"relative_path": "workspace/problem.md"},
                                call_id="call-b",
                            ),
                        ]
                    )
                },
                {
                    "response": ProviderResponse(content="Both tools ran."),
                },
            ]
        )
        self.app.agent.provider = provider

        events = list(self.app.ask_stream("Call two tools.", self.state))
        self.assertEqual(events[-1].text, "Both tools ran.")
        self._assert_no_orphan_tool_calls_across_provider_calls(provider)


if __name__ == "__main__":
    unittest.main()

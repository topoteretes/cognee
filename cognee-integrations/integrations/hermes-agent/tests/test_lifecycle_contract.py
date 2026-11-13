"""Characterization: the Hermes-only lifecycle semantics.

None of this exists in the claude-code / codex / openclaw plugins, so there is
nothing to copy it from — it can only be preserved deliberately. Covered here:

* ``agent_context`` write gating (subagents must not write)
* the ``queue_prefetch`` → ``prefetch`` two-phase protocol and its formatting
* ``sync_turn`` turn framing and session-cache routing
* ``on_memory_write`` mirroring, ``on_delegation``, ``on_session_switch``
* session-id derivation and the in-memory circuit breaker

Run standalone with ``python3 tests/test_lifecycle_contract.py``.
"""

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import FakeBackend, fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes import provider as provider_mod  # noqa: E402

_NO_URL = {"COGNEE_BASE_URL": "", "COGNEE_SERVICE_URL": ""}

# How long to wait for a call that should never arrive. The write paths spawn
# daemon workers that reach the (inline) fake in microseconds, so this is ~1000x
# margin while keeping the suite fast.
_NO_CALL_GRACE = 0.2


def assert_no_call(case, fake, name):
    """Assert *name* is never called, having waited long enough to mean it.

    Asserting immediately after a suppressed write would pass even if the guard
    were missing — the worker simply would not have got there yet. Worse, that
    worker would outlive the ``fake_backend`` context and import the *real*
    cognee, racing the next test's fake. Waiting for the call not to arrive
    fixes both, so this must be called *inside* the ``fake_backend`` block.
    """
    case.assertFalse(
        fake.wait(name, timeout=_NO_CALL_GRACE),
        f"expected no {name!r} call, but one was made",
    )
    case.assertEqual(fake.kwargs_for(name), [])


def _settle_prefetch(provider, timeout=2.0):
    """Wait for a queued prefetch to finish *writing its result*.

    ``fake.wait("recall")`` only proves the backend was called; the worker stores
    the formatted result after that call returns. Joining the worker is the
    deterministic barrier.
    """
    thread = provider._prefetch_thread
    if thread is not None:
        thread.join(timeout=timeout)


def _initialize(backend=None, **init_kwargs):
    """Run the real ``initialize()`` in embedded mode against a fake transport.

    Embedded mode is the only path that needs neither a server nor the network:
    it configures local roots and resolves an identity, both of which the fake
    backend records without touching anything.
    """
    provider = provider_mod.CogneeMemoryProvider(backend=backend or FakeBackend())
    env = {**_NO_URL, "COGNEE_EMBEDDED": "true"}
    with mock.patch.dict("os.environ", env, clear=False):
        provider.initialize("sess-1", **init_kwargs)
    return provider


# --------------------------------------------------------------------------
# Write gating — subagents must not write to memory
# --------------------------------------------------------------------------


class TestAgentContextWriteGating(unittest.TestCase):
    def test_absent_agent_context_enables_writes(self):
        with fake_backend():
            self.assertTrue(_initialize()._writes_enabled)

    def test_primary_empty_and_none_enable_writes(self):
        with fake_backend():
            for value in ("primary", "", None):
                self.assertTrue(
                    _initialize(agent_context=value)._writes_enabled,
                    f"agent_context={value!r} should allow writes",
                )

    def test_subagent_context_disables_writes(self):
        with fake_backend():
            self.assertFalse(_initialize(agent_context="subagent")._writes_enabled)

    def test_disabled_writes_suppress_sync_turn(self):
        with fake_backend() as fake:
            provider = make_provider(writes_enabled=False)
            provider.sync_turn("u", "a")
            assert_no_call(self, fake, "remember_permanent")

    def test_disabled_writes_suppress_session_end_improve(self):
        with fake_backend() as fake:
            provider = make_provider(writes_enabled=False)
            provider.on_session_end([])
        self.assertEqual(fake.kwargs_for("improve"), [])

    def test_disabled_writes_suppress_the_memory_write_mirror(self):
        # Regression: on_memory_write used to check only the breaker, so a
        # subagent's built-in memory write reached the graph even though its
        # conversation turns were suppressed. Write gating is now uniform.
        with fake_backend() as fake:
            provider = make_provider(writes_enabled=False)
            provider.on_memory_write("add", "project", "note")
            assert_no_call(self, fake, "remember_permanent")

    def test_enabled_writes_still_mirror(self):
        with fake_backend() as fake:
            provider = make_provider(writes_enabled=True)
            provider.on_memory_write("add", "project", "note")
            self.assertTrue(fake.wait("remember_permanent"))
        self.assertEqual(len(fake.kwargs_for("remember_permanent")), 1)


# --------------------------------------------------------------------------
# Fail closed — a provider that never initialized must not touch cognee
# --------------------------------------------------------------------------


class TestFailsClosedWhenUninitialized(unittest.TestCase):
    """Hermes swallows whatever ``initialize()`` raises and starts anyway.

    ``MemoryManager.initialize_all`` logs the exception and continues, leaving the
    provider registered with its tool schemas live. Every entry point therefore has
    to check for itself, or calls would run against an unconnected transport — for
    the SDK that means silently doing in-process cognee, the exact single-writer DB
    risk local-server mode exists to prevent.
    """

    def _uninitialized(self, fake):
        provider = make_provider(backend=fake)
        provider._initialized = False  # what a failed initialize() leaves behind
        return provider

    def test_a_failed_initialize_leaves_the_provider_unusable(self):
        with fake_backend() as fake:
            fake.errors["connect"] = RuntimeError("unreachable")
            provider = provider_mod.CogneeMemoryProvider(backend=fake)
            env = {**_NO_URL, "COGNEE_BASE_URL": "https://cloud.example"}
            with mock.patch.dict("os.environ", env, clear=False):
                with self.assertRaises(RuntimeError):
                    provider.initialize("sid")
            self.assertFalse(provider._initialized)

    def test_tool_calls_report_unavailable_instead_of_running(self):
        with fake_backend() as fake:
            provider = self._uninitialized(fake)
            for tool, args in (
                ("cognee_recall", {"query": "q"}),
                ("cognee_remember", {"content": "c"}),
                ("cognee_forget", {"everything": True}),
            ):
                out = json.loads(provider.handle_tool_call(tool, args))
                self.assertIn("unavailable", out["error"])
            self.assertEqual(fake.calls, [])

    def test_writes_are_suppressed(self):
        with fake_backend() as fake:
            provider = self._uninitialized(fake)
            provider.sync_turn("u", "a")
            provider.on_memory_write("add", "project", "note")
            provider.on_delegation("task", "result")
            provider.on_session_end([])
            assert_no_call(self, fake, "remember_session")
            assert_no_call(self, fake, "remember_permanent")
            self.assertEqual(fake.kwargs_for("improve"), [])

    def test_recall_is_not_queued_or_returned(self):
        with fake_backend() as fake:
            provider = self._uninitialized(fake)
            provider.queue_prefetch("q")
            assert_no_call(self, fake, "recall")
            self.assertEqual(provider.prefetch("q"), "")

    def test_memory_is_not_advertised_to_the_model(self):
        with fake_backend() as fake:
            self.assertEqual(self._uninitialized(fake).system_prompt_block(), "")

    def test_an_initialized_provider_still_advertises_memory(self):
        with fake_backend():
            block = make_provider().system_prompt_block()
        self.assertIn("Cognee Memory", block)
        self.assertIn("cognee_recall", block)


# --------------------------------------------------------------------------
# sync_turn
# --------------------------------------------------------------------------


class TestSyncTurn(unittest.TestCase):
    def test_turn_framing_and_session_cache_routing(self):
        with fake_backend() as fake:
            provider = make_provider(session_cognee_id="hermes_s1")
            provider.sync_turn("what is up", "not much")
            self.assertTrue(fake.wait("remember_session"))
            kwargs = fake.only_call("remember_session")
        self.assertEqual(kwargs["text"], "User: what is up\nAssistant: not much")
        self.assertEqual(kwargs["session_id"], "hermes_s1")
        self.assertEqual(kwargs["dataset"], "hermes")
        # Routing to the session cache is expressed by the method chosen, so the
        # permanent-graph path must not also have fired.
        self.assertEqual(fake.kwargs_for("remember_permanent"), [])

    def test_open_breaker_suppresses_the_write(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider._is_breaker_open = lambda: True
            provider.sync_turn("u", "a")
            assert_no_call(self, fake, "remember_session")

    def test_per_call_session_id_overrides_the_initialized_session(self):
        with fake_backend() as fake:
            provider = make_provider(session_id="s-1", session_cognee_id="hermes_s-1")
            provider.sync_turn("u", "a", session_id="s-2")
            self.assertTrue(fake.wait("remember_session"))
            self.assertEqual(fake.only_call("remember_session")["session_id"], "hermes_s-2")

    def test_matching_session_id_uses_the_cached_derived_id(self):
        with fake_backend() as fake:
            provider = make_provider(session_id="s-1", session_cognee_id="custom_id")
            provider.sync_turn("u", "a", session_id="s-1")
            self.assertTrue(fake.wait("remember_session"))
            self.assertEqual(fake.only_call("remember_session")["session_id"], "custom_id")


# --------------------------------------------------------------------------
# prefetch / queue_prefetch two-phase protocol
# --------------------------------------------------------------------------


def _graph_prefetch_call(provider, fake):
    """The kwargs of one prefetch's graph request, once every lane has run."""
    provider.queue_prefetch("q")
    provider._prefetch_thread.join(timeout=5.0)
    calls = [kwargs for kwargs in fake.kwargs_for("recall") if kwargs["scope"] == ["graph"]]
    if len(calls) != 1:
        raise AssertionError(f"expected exactly 1 graph recall, got {len(calls)}")
    return calls[0]


class TestPrefetchProtocol(unittest.TestCase):
    """The two-phase protocol around the per-prompt memory block; the block's
    own request and rendering are characterized in test_layered_recall."""

    def test_queued_result_is_returned_with_the_cognee_memory_header(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "remembered thing", "source": "graph"}]
            provider = make_provider()
            provider.queue_prefetch("q")
            out = provider.prefetch("q")
        self.assertEqual(
            out, "## Cognee Memory\n<cognee_memory>\nremembered thing\n</cognee_memory>"
        )

    def test_prefetch_without_a_queued_result_is_empty(self):
        with fake_backend():
            provider = make_provider()
            self.assertEqual(provider.prefetch("q"), "")

    def test_prefetch_consumes_the_cached_result(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "once"}]
            provider = make_provider()
            provider.queue_prefetch("q")
            self.assertNotEqual(provider.prefetch("q"), "")
            self.assertEqual(provider.prefetch("q"), "")

    def test_empty_recall_leaves_nothing_cached(self):
        with fake_backend() as fake:
            fake.results["recall"] = []
            provider = make_provider()
            provider.queue_prefetch("q")
            self.assertEqual(provider.prefetch("q"), "")

    def test_blank_query_is_not_queued(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.queue_prefetch("")
            assert_no_call(self, fake, "recall")

    def test_open_breaker_is_not_queued(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider._is_breaker_open = lambda: True
            provider.queue_prefetch("q")
            assert_no_call(self, fake, "recall")

    def test_prefetch_uses_a_smaller_budget_than_explicit_recall(self):
        with fake_backend() as fake:
            provider = make_provider(top_k=12)
            kwargs = _graph_prefetch_call(provider, fake)
        self.assertEqual(kwargs["top_k"], 5)
        # The graph request carries no session id (SDK-904); the guidance lane does.
        self.assertIsNone(kwargs["session_id"])
        # Auto recall (SDK-1393): unpinned, the graph lane names no dataset.
        self.assertIsNone(kwargs["datasets"])
        self.assertEqual(kwargs["scope"], ["graph"])
        self.assertEqual(kwargs["query_type"], "HYBRID_COMPLETION")
        self.assertIs(kwargs["only_context"], True)

    def test_there_is_only_one_prefetch_path(self):
        # The legacy single auto-scope search behind ``recall_session_layers``
        # is gone: the toggle is not read, and the request is the graph one.
        with fake_backend() as fake:
            provider = make_provider(config={"recall_session_layers": False})
            kwargs = _graph_prefetch_call(provider, fake)
        self.assertEqual(kwargs["scope"], ["graph"])
        self.assertIs(kwargs["only_context"], True)

    def test_prefetch_budget_respects_a_lower_configured_top_k(self):
        with fake_backend() as fake:
            provider = make_provider(top_k=2)
            self.assertEqual(_graph_prefetch_call(provider, fake)["top_k"], 2)

    def test_every_memory_item_is_injected_whole(self):
        # No line cap and no 500-character cut: the item's text is the memory.
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": f"item {i} " + "x" * 900} for i in range(8)]
            provider = make_provider()
            provider.queue_prefetch("q")
            out = provider.prefetch("q")
        self.assertEqual(out.count("x"), 8 * 900)
        for i in range(8):
            self.assertIn(f"item {i} ", out)

    def test_blank_text_results_are_skipped(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "   "}, {"text": "kept"}]
            provider = make_provider()
            provider.queue_prefetch("q")
            out = provider.prefetch("q")
        self.assertEqual(out, "## Cognee Memory\n<cognee_memory>\nkept\n</cognee_memory>")

    def test_recall_failure_leaves_prefetch_empty_and_does_not_raise(self):
        with fake_backend() as fake:
            fake.errors["recall"] = RuntimeError("down")
            provider = make_provider()
            provider.queue_prefetch("q")
            self.assertEqual(provider.prefetch("q"), "")


# --------------------------------------------------------------------------
# on_session_end / on_memory_write / on_delegation / on_session_switch
# --------------------------------------------------------------------------


class TestSessionEnd(unittest.TestCase):
    def test_improve_targets_the_dataset_and_the_session(self):
        with fake_backend() as fake:
            provider = make_provider(session_cognee_id="hermes_sX")
            provider.on_session_end([])
            kwargs = fake.only_call("improve")
        self.assertEqual(kwargs["dataset"], "hermes")
        self.assertEqual(kwargs["session_ids"], ["hermes_sX"])
        self.assertIn("background", kwargs)

    def test_improve_on_end_disabled_skips_it(self):
        with fake_backend() as fake:
            provider = make_provider(improve_on_end=False)
            provider.on_session_end([])
        self.assertEqual(fake.kwargs_for("improve"), [])

    def test_open_breaker_skips_it(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider._is_breaker_open = lambda: True
            provider.on_session_end([])
        self.assertEqual(fake.kwargs_for("improve"), [])

    def _background_for(self, **kwargs):
        with fake_backend() as fake:
            make_provider(**kwargs).on_session_end([])
            return fake.only_call("improve")["background"]

    def test_an_armed_session_end_hands_the_close_to_a_detached_worker(self):
        # The headline guarantee: the user's exit does not wait on a graph build.
        # Nothing is improved in this process — the detached worker does it, and
        # only it may unregister afterwards.
        with fake_backend() as fake:
            provider = make_provider(
                watcher_state_path=Path("/tmp/w.json"), session_cognee_id="hermes_sZ"
            )
            with mock.patch.object(
                provider_mod.exit_watcher, "finalize", return_value=True
            ) as finalize:
                provider.on_session_end([])
            self.assertEqual(fake.kwargs_for("improve"), [])
        kwargs = finalize.call_args.kwargs
        self.assertEqual(kwargs["session_id"], "hermes_sZ")
        self.assertEqual(kwargs["dataset"], "hermes")
        self.assertTrue(kwargs["improve"])
        self.assertTrue(provider._close_handed_off)

    def test_a_failed_handoff_falls_back_to_an_inline_synchronous_improve(self):
        # shutdown() is about to unregister and take the server with it, so the
        # bridge has to finish here or not at all.
        with fake_backend() as fake:
            provider = make_provider(watcher_state_path=Path("/tmp/w.json"))
            with mock.patch.object(provider_mod.exit_watcher, "finalize", return_value=False):
                provider.on_session_end([])
            self.assertIs(fake.only_call("improve")["background"], False)
        self.assertFalse(provider._close_handed_off)

    def test_no_watcher_means_an_inline_synchronous_improve(self):
        # Embedded mode: the work runs in this process and dies with it, so there
        # is nobody to hand off to and no server to protect.
        self.assertFalse(self._background_for(remote_mode=False))

    def test_the_override_opts_out_of_the_handoff_entirely(self):
        for value, expected in (("true", True), ("false", False)):
            with fake_backend() as fake:
                provider = make_provider(
                    watcher_state_path=Path("/tmp/w.json"),
                    config={"improve_background": value},
                )
                with mock.patch.object(provider_mod.exit_watcher, "finalize") as finalize:
                    provider.on_session_end([])
                finalize.assert_not_called()
                self.assertIs(fake.only_call("improve")["background"], expected)


class TestMemoryWriteMirror(unittest.TestCase):
    def _mirror(self, *args, **kwargs):
        with fake_backend() as fake:
            provider = make_provider(session_cognee_id="hermes_sM")
            provider.on_memory_write(*args, **kwargs)
            if not fake.wait("remember_permanent", timeout=2.0):
                return fake, None
            return fake, fake.only_call("remember_permanent")

    def test_payload_format_with_the_default_origin(self):
        _, kwargs = self._mirror("add", "project", "prefers tabs")
        self.assertEqual(
            kwargs["text"],
            "Hermes project memory (add, hermes_memory_tool): prefers tabs",
        )

    def test_write_origin_metadata_replaces_the_default_origin(self):
        _, kwargs = self._mirror("replace", "user", "likes dark mode", {"write_origin": "user_md"})
        self.assertEqual(
            kwargs["text"],
            "Hermes user memory (replace, user_md): likes dark mode",
        )

    def test_mirror_targets_the_permanent_graph(self):
        _, kwargs = self._mirror("add", "project", "note")
        self.assertEqual(kwargs["session_ids"], ["hermes_sM"])
        self.assertEqual(kwargs["dataset"], "hermes")

    def test_only_add_and_replace_are_mirrored(self):
        with fake_backend() as fake:
            provider = make_provider()
            for action in ("delete", "remove", "read", ""):
                provider.on_memory_write(action, "project", "note")
            assert_no_call(self, fake, "remember_permanent")

    def test_empty_content_is_not_mirrored(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.on_memory_write("add", "project", "")
            assert_no_call(self, fake, "remember_permanent")

    def test_open_breaker_suppresses_the_mirror(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider._is_breaker_open = lambda: True
            provider.on_memory_write("add", "project", "note")
            assert_no_call(self, fake, "remember_permanent")


class TestDelegation(unittest.TestCase):
    def test_parent_records_task_and_result_as_a_turn(self):
        with fake_backend() as fake:
            provider = make_provider(session_cognee_id="hermes_parent")
            provider.on_delegation("do the thing", "did the thing", child_session_id="child-1")
            self.assertTrue(fake.wait("remember_session"))
            kwargs = fake.only_call("remember_session")
        self.assertEqual(
            kwargs["text"],
            "User: Delegated task: do the thing\nResult: did the thing\nAssistant: ",
        )
        self.assertEqual(kwargs["session_id"], "hermes_parent")

    def test_empty_task_and_result_records_nothing(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.on_delegation("", "")
            assert_no_call(self, fake, "remember_session")


class TestSessionSwitch(unittest.TestCase):
    def test_switch_rederives_the_session_id(self):
        with fake_backend():
            provider = make_provider(session_id="s-1")
            provider.on_session_switch("s-2")
        self.assertEqual(provider._session_id, "s-2")
        self.assertEqual(provider._session_cognee_id, "hermes_s-2")

    def test_reset_clears_a_settled_prefetch(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "stale"}]
            provider = make_provider()
            provider.queue_prefetch("q")
            _settle_prefetch(provider)
            provider.on_session_switch("s-2", reset=True)
            self.assertEqual(provider.prefetch("q"), "")

    def test_non_reset_switch_keeps_a_settled_prefetch(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "carried over"}]
            provider = make_provider()
            provider.queue_prefetch("q")
            _settle_prefetch(provider)
            provider.on_session_switch("s-2", parent_session_id="s-1", reset=False)
            self.assertIn("carried over", provider.prefetch("q"))

    def test_reset_discards_a_prefetch_that_was_in_flight(self):
        """Regression: no cross-conversation context leak on ``/reset``.

        A recall issued for the previous conversation must not land in the fresh
        one, even when it returns *after* the reset cleared the slot. Held inside
        the backend call with a gate so the in-flight window is deterministic
        rather than timing-dependent.
        """
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "belongs to the old conversation"}]
            gate = threading.Event()
            fake.gates["recall"] = gate

            provider = make_provider()
            provider.queue_prefetch("q")
            self.assertTrue(fake.wait("recall"))  # worker is now inside the recall

            provider.on_session_switch("s-2", reset=True)
            gate.set()  # let the recall return, after the reset
            _settle_prefetch(provider)

            self.assertEqual(provider.prefetch("q"), "")

    def test_non_reset_switch_keeps_a_prefetch_that_was_in_flight(self):
        """The mirror case: /resume, /branch and compression continue the same
        logical conversation, so an in-flight prefetch stays valid."""
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "still relevant"}]
            gate = threading.Event()
            fake.gates["recall"] = gate

            provider = make_provider()
            provider.queue_prefetch("q")
            self.assertTrue(fake.wait("recall"))

            provider.on_session_switch("s-2", parent_session_id="s-1", reset=False)
            gate.set()
            _settle_prefetch(provider)

            self.assertIn("still relevant", provider.prefetch("q"))


# --------------------------------------------------------------------------
# Session-id derivation — drift here silently orphans history
# --------------------------------------------------------------------------


class TestSessionIdDerivation(unittest.TestCase):
    def test_default_prefix(self):
        provider = make_provider()
        self.assertEqual(provider._build_cognee_session_id("abc123"), "hermes_abc123")

    def test_configured_prefix(self):
        provider = make_provider(config={"session_prefix": "hermes-cli"})
        self.assertEqual(provider._build_cognee_session_id("abc"), "hermes-cli_abc")

    def test_extra_kwargs_are_accepted_but_not_embedded(self):
        provider = make_provider()
        derived = provider._build_cognee_session_id("abc", agent_workspace="/tmp/ws", user_id="u-9")
        self.assertEqual(derived, "hermes_abc")

    def test_allowed_characters_survive(self):
        self.assertEqual(
            provider_mod._safe_session_component("Ab9-c_d.e"),
            "Ab9-c_d.e",
        )

    def test_disallowed_characters_become_underscores(self):
        self.assertEqual(provider_mod._safe_session_component("a/b c:d"), "a_b_c_d")

    def test_leading_and_trailing_dots_and_underscores_are_stripped(self):
        self.assertEqual(provider_mod._safe_session_component("._abc_."), "abc")

    def test_capped_at_120_characters(self):
        self.assertEqual(len(provider_mod._safe_session_component("x" * 200)), 120)

    def test_empty_input_falls_back_to_session(self):
        self.assertEqual(provider_mod._safe_session_component(""), "session")
        self.assertEqual(provider_mod._safe_session_component("..."), "session")

    def test_session_cognee_id_for_blank_returns_the_current_session(self):
        provider = make_provider(session_id="s-1", session_cognee_id="cached")
        self.assertEqual(provider._session_cognee_id_for(""), "cached")

    def test_session_cognee_id_for_current_session_returns_the_cached_id(self):
        provider = make_provider(session_id="s-1", session_cognee_id="cached")
        self.assertEqual(provider._session_cognee_id_for("s-1"), "cached")

    def test_session_cognee_id_for_another_session_is_derived(self):
        provider = make_provider(session_id="s-1", session_cognee_id="cached")
        self.assertEqual(provider._session_cognee_id_for("s-2"), "hermes_s-2")


# --------------------------------------------------------------------------
# Circuit breaker — in-memory on purpose (long-lived provider process)
# --------------------------------------------------------------------------


class TestCircuitBreaker(unittest.TestCase):
    def test_stays_closed_below_the_threshold(self):
        provider = make_provider()
        for _ in range(4):
            provider._record_failure()
        self.assertFalse(provider._is_breaker_open())

    def test_opens_at_the_threshold(self):
        provider = make_provider()
        for _ in range(5):
            provider._record_failure()
        self.assertTrue(provider._is_breaker_open())

    def test_success_resets_the_failure_count(self):
        provider = make_provider()
        for _ in range(4):
            provider._record_failure()
        provider._record_success()
        provider._record_failure()
        self.assertFalse(provider._is_breaker_open())

    def test_closes_again_once_the_cooldown_expires(self):
        provider = make_provider()
        for _ in range(5):
            provider._record_failure()
        provider._breaker_open_until = time.monotonic() - 1
        self.assertFalse(provider._is_breaker_open())
        self.assertEqual(provider._consecutive_failures, 0)

    def test_threshold_and_cooldown_constants(self):
        self.assertEqual(provider_mod._BREAKER_THRESHOLD, 5)
        self.assertEqual(provider_mod._BREAKER_COOLDOWN_SECS, 120)

    def test_concurrent_failures_are_not_lost(self):
        # The breaker is hit from the main thread and every worker thread; an
        # unlocked `+= 1` can drop increments under that concurrency, delaying
        # the trip. Hammer it and require an exact count.
        provider = make_provider()
        per_thread = 200

        def fail_repeatedly():
            for _ in range(per_thread):
                provider._record_failure()

        threads = [threading.Thread(target=fail_repeatedly) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(provider._consecutive_failures, 8 * per_thread)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""The per-prompt memory block and the memory-hit header.

Every prompt makes one memory request — ``scope=["graph"]``,
``query_type=HYBRID_COMPLETION``, ``only_context=True`` and NO session id —
one guidance request (``scope=["session_context"]`` with this conversation's
session id), plus the deterministic code lane when it is armed (SDK-904, the
openclaw parity contract). On cognee >= 1.6.0 the graph item's ``text`` is the
full LLM input the completion would have received; it is trimmed to the
retrieved context (``recall_text.trim_recall_text``): the question template and
history go, this session's own passages go, passages are capped, and
``recall_context_chars`` caps the whole injection. Text that does not parse (an
older server's bare context) is injected whole. Run standalone with
``python3 tests/test_layered_recall.py``.
"""

import asyncio
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import fake_backend, fake_cognee, make_provider  # noqa: E402
from cognee_integration_hermes import provider as provider_mod  # noqa: E402
from cognee_integration_hermes.backend import SdkBackend  # noqa: E402
from cognee_integration_hermes.recall_text import TRUNCATION_MARK  # noqa: E402

_LAYERED = {"recall_budget": 20}
# Arms the code lane: a configured code dataset plus an identifier in the prompt.
_WITH_CODE = {**_LAYERED, "code_datasets": "codebase-svc-abc123"}
_CODE_QUERY = "what calls process_payment?"
_QUERY = "what database does the ledger use?"

# A cognee >= 1.6.0 only_context completion item: ``text`` is the whole prompt
# the completion would have read, ``system_prompt`` the retriever's task
# template. History first, the question template around the context, and the
# session guidance block at the end.
_SYSTEM_PROMPT = "Answer the question using the provided context."
_HISTORY = "Previous conversation:\n\nQUESTION: which db?\nANSWER: we picked Postgres."
_OTHER_PASSAGE = "Session ID: claude_other\n\nQuestion: ledger?\n\nAnswer: Postgres stores it."
_OWN_PASSAGE = "Session ID: hermes_s-1\n\nQuestion: which db?\n\nAnswer: Postgres, per this chat."
_CONTEXT = f"## Relevant passages\n{_OTHER_PASSAGE}\n---\n{_OWN_PASSAGE}"
_ITEM_GUIDANCE = "## Active session guidance\n- prefer concise answers"
_PROMPT_TEXT = (
    f"{_HISTORY}\n\nThe question is: `{_QUERY}`\n"
    f"Answer using this context:\n`{_CONTEXT}`\n\n{_ITEM_GUIDANCE}"
)
_V160_ITEM = {"source": "graph", "text": _PROMPT_TEXT, "system_prompt": _SYSTEM_PROMPT}
_OLD_SERVER_ITEM = {"source": "graph", "text": "bare context"}


def _settle(provider, timeout=5.0):
    thread = provider._prefetch_thread
    if thread is not None:
        thread.join(timeout=timeout)


def _prefetch(provider, query="q"):
    provider.queue_prefetch(query)
    _settle(provider)
    return provider.prefetch(query)


def _block(out, tag):
    """The rendered content between ``<tag>`` and ``</tag>``."""
    start = out.index(f"<{tag}>\n") + len(f"<{tag}>\n")
    end = out.index(f"\n</{tag}>")
    return out[start:end]


def _by_scope(calls, scope):
    return [kwargs for kwargs in calls if kwargs["scope"] == [scope]]


class TestPerPromptRequests(unittest.TestCase):
    def test_one_graph_and_one_guidance_request_per_prompt(self):
        with fake_backend() as fake:
            provider = make_provider(config=_LAYERED)
            _prefetch(provider)
            calls = fake.kwargs_for("recall")
        self.assertEqual(
            sorted(tuple(c["scope"]) for c in calls), [("graph",), ("session_context",)]
        )

    def test_the_graph_request_is_explicit_and_carries_no_session(self):
        with fake_backend() as fake:
            provider = make_provider(config=_LAYERED)
            _prefetch(provider)
            (kwargs,) = _by_scope(fake.kwargs_for("recall"), "graph")
        self.assertEqual(kwargs["query_type"], "HYBRID_COMPLETION")
        self.assertIs(kwargs["only_context"], True)
        # Without a session id the server does not put this conversation's own
        # turns into the item: Hermes already has them.
        self.assertIsNone(kwargs["session_id"])
        # Auto recall (SDK-1393): unpinned, it names no dataset and the server picks.
        self.assertIsNone(kwargs["datasets"])
        self.assertIsNone(kwargs["code_query"])

    def test_the_guidance_request_carries_the_session_and_the_capture_dataset(self):
        with fake_backend() as fake:
            provider = make_provider(config=_LAYERED)
            _prefetch(provider)
            (kwargs,) = _by_scope(fake.kwargs_for("recall"), "session_context")
        self.assertEqual(kwargs["session_id"], "hermes_s-1")
        self.assertEqual(kwargs["datasets"], ["hermes"])
        self.assertIsNone(kwargs["query_type"])
        self.assertIs(kwargs["only_context"], True)

    def test_session_cache_scopes_are_never_requested(self):
        with fake_backend() as fake:
            provider = make_provider(config=_LAYERED)
            _prefetch(provider)
            scopes = [tuple(kwargs["scope"]) for kwargs in fake.kwargs_for("recall")]
        for retired in (("session",), ("trace",), ("auto",)):
            self.assertNotIn(retired, scopes)

    def test_the_code_lane_joins_when_armed(self):
        with fake_backend() as fake:
            provider = make_provider(config=_WITH_CODE)
            _prefetch(provider, _CODE_QUERY)
            calls = fake.kwargs_for("recall")
        self.assertEqual(
            sorted(tuple(c["scope"]) for c in calls),
            [("code",), ("graph",), ("session_context",)],
        )
        for kwargs in calls:
            self.assertIs(kwargs["only_context"], True)

    def test_zero_budget_skips_the_request(self):
        with fake_backend() as fake:
            provider = make_provider(config={**_LAYERED, "recall_budget": 0})
            _prefetch(provider)
            self.assertEqual(fake.kwargs_for("recall"), [])


class TestTrimming(unittest.TestCase):
    def test_the_v160_item_is_cut_to_the_retrieved_context(self):
        with fake_backend() as fake:
            fake.results["recall"] = [_V160_ITEM]
            provider = make_provider(config={**_LAYERED, "memory_hits": False})
            out = _prefetch(provider, _QUERY)
        memory = _block(out, "cognee_memory")
        self.assertIn(_OTHER_PASSAGE, memory)
        # The history, the question template and this session's own passage go.
        self.assertNotIn("Previous conversation", out)
        self.assertNotIn("The question is", out)
        self.assertNotIn("per this chat", out)

    def test_item_guidance_is_injected_as_its_own_block(self):
        with fake_backend() as fake:
            fake.results["recall"] = [_V160_ITEM]
            provider = make_provider(config={**_LAYERED, "memory_hits": False})
            out = _prefetch(provider, _QUERY)
        self.assertIn("prefer concise answers", _block(out, "cognee_guidance"))
        self.assertNotIn("prefer concise answers", _block(out, "cognee_memory"))

    def test_the_guidance_lane_wins_over_the_items_guidance(self):
        with fake_backend() as fake:
            fake.results["recall"] = [_V160_ITEM]
            fake.results["recall:session_context"] = [{"text": "- lesson from the server"}]
            provider = make_provider(config={**_LAYERED, "memory_hits": False})
            out = _prefetch(provider, _QUERY)
        self.assertEqual(_block(out, "cognee_guidance"), "- lesson from the server")

    def test_guidance_alone_is_injected(self):
        with fake_backend() as fake:
            fake.results["recall:session_context"] = [{"text": "- a lesson"}]
            provider = make_provider(config={**_LAYERED, "memory_hits": False})
            out = _prefetch(provider)
        self.assertEqual(out, "## Cognee Memory\n<cognee_guidance>\n- a lesson\n</cognee_guidance>")

    def test_system_prompt_is_never_injected(self):
        with fake_backend() as fake:
            fake.results["recall"] = [_V160_ITEM]
            provider = make_provider(config=_LAYERED)
            out = _prefetch(provider, _QUERY)
        self.assertNotIn(_SYSTEM_PROMPT, out)
        self.assertNotIn("system_prompt", out)

    def test_old_server_bare_context_renders_whole(self):
        with fake_backend() as fake:
            fake.results["recall"] = [_OLD_SERVER_ITEM]
            provider = make_provider(config={**_LAYERED, "memory_hits": False})
            out = _prefetch(provider)
        self.assertEqual(out, "## Cognee Memory\n<cognee_memory>\nbare context\n</cognee_memory>")

    def test_passages_are_capped(self):
        long_passage = "Session ID: claude_x\n\n" + "\n\n".join(
            f"para {i} " * 20 for i in range(40)
        )
        text = f"The question is: `{_QUERY}`\n`## Relevant passages\n{long_passage}`"
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": text}]
            provider = make_provider(config={**_LAYERED, "recall_passage_chars": 500})
            out = _prefetch(provider, _QUERY)
        memory = _block(out, "cognee_memory")
        self.assertIn("…[+", memory)
        self.assertLess(len(memory), 700)

    def test_the_whole_injection_is_capped(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "line\n" * 3000}]
            provider = make_provider(config={**_LAYERED, "recall_context_chars": 1000})
            out = _prefetch(provider)
        memory = _block(out, "cognee_memory")
        self.assertTrue(memory.endswith(TRUNCATION_MARK.strip("\n")))
        self.assertLessEqual(len(memory), 1000)

    def test_zero_disables_the_caps(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"source": "graph", "text": "x" * 20000}]
            provider = make_provider(
                config={**_LAYERED, "recall_context_chars": 0, "recall_passage_chars": 0}
            )
            out = _prefetch(provider)
        self.assertEqual(out.count("x"), 20000)

    def test_text_wins_over_other_fields(self):
        with fake_backend() as fake:
            fake.results["recall"] = [
                {
                    "source": "graph",
                    "text": "the memory",
                    "content": "not this",
                    "chunk_text": "nor this",
                }
            ]
            provider = make_provider(config={**_LAYERED, "memory_hits": False})
            out = _prefetch(provider)
        self.assertEqual(_block(out, "cognee_memory"), "the memory")

    def test_blank_items_leave_nothing_cached(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"source": "graph", "text": "   "}]
            provider = make_provider(config=_LAYERED)
            self.assertEqual(_prefetch(provider), "")

    def test_empty_recall_leaves_nothing_cached(self):
        with fake_backend() as fake:
            fake.results["recall"] = []
            provider = make_provider(config=_LAYERED)
            self.assertEqual(_prefetch(provider), "")

    def test_blocks_render_in_canonical_order_whatever_answers_first(self):
        with fake_backend() as fake:
            original = fake.recall

            def staggered(**kwargs):
                # The memory lane answers last; the rendering order must not care.
                time.sleep(0.25 if kwargs["scope"] == ["graph"] else 0.0)
                original(**kwargs)
                return [{"text": f"from {kwargs['scope'][0]}"}]

            fake.recall = staggered
            provider = make_provider(config={**_WITH_CODE, "memory_hits": False})
            out = _prefetch(provider, _CODE_QUERY)
        self.assertLess(out.index("<cognee_memory>"), out.index("<code_graph>"), out)
        self.assertLess(out.index("<code_graph>"), out.index("<cognee_guidance>"), out)
        self.assertIn("- [cognee] from code", _block(out, "code_graph"))
        self.assertEqual(_block(out, "cognee_memory"), "from graph")
        self.assertEqual(_block(out, "cognee_guidance"), "from session_context")


class TestLaneResilience(unittest.TestCase):
    def test_lanes_run_concurrently_under_one_deadline(self):
        # Three lanes sleeping 0.3s each must cost ~0.3s, not 0.9s, and all must
        # be handed the same deadline: min(recall_timeout, budget).
        with fake_backend() as fake:
            original = fake.recall

            def slow(**kwargs):
                time.sleep(0.3)
                return original(**kwargs)

            fake.recall = slow
            provider = make_provider(config={**_WITH_CODE, "recall_timeout": 5, "recall_budget": 2})
            started = time.monotonic()
            _prefetch(provider, _CODE_QUERY)
            wall = time.monotonic() - started
            timeouts = {kwargs["timeout"] for kwargs in fake.kwargs_for("recall")}
        self.assertEqual(len(fake.kwargs_for("recall")), 3)
        self.assertLess(wall, 0.55, f"lanes ran back to back: {wall:.2f}s for 3 x 0.3s")
        self.assertEqual(len(timeouts), 1, timeouts)
        self.assertLessEqual(max(timeouts), 2.0)
        self.assertGreater(max(timeouts), 1.5)

    def test_a_failing_code_lane_does_not_discard_the_memory(self):
        with fake_backend() as fake:
            original = fake.recall

            def flaky(**kwargs):
                if kwargs["scope"] == ["code"]:
                    raise RuntimeError("code lane down")
                original(**kwargs)
                return [_OLD_SERVER_ITEM]

            fake.recall = flaky
            provider = make_provider(config={**_WITH_CODE, "memory_hits": False})
            out = _prefetch(provider, _CODE_QUERY)
        self.assertIn("<cognee_memory>", out)
        self.assertNotIn("<code_graph>", out)
        self.assertEqual(provider._consecutive_failures, 0)

    def test_a_failing_guidance_lane_keeps_the_memory_and_the_breaker_quiet(self):
        with fake_backend() as fake:
            fake.results["recall"] = [_OLD_SERVER_ITEM]
            fake.errors["recall:session_context"] = RuntimeError("guidance down")
            provider = make_provider(config={**_LAYERED, "memory_hits": False})
            out = _prefetch(provider)
        self.assertIn("<cognee_memory>", out)
        self.assertNotIn("<cognee_guidance>", out)
        self.assertEqual(provider._consecutive_failures, 0)

    def test_a_graph_404_is_benign_not_a_breaker_failure(self):
        # A dataset nobody has cognified answers the graph scope with 404 on
        # every prompt of a fresh install; that must not feed the breaker.
        class _Http404(RuntimeError):
            status = 404

        with fake_backend() as fake:
            original = fake.recall

            def not_built(**kwargs):
                original(**kwargs)
                if kwargs["scope"] in (["graph"], ["session_context"]):
                    raise _Http404("nothing yet")
                return []

            fake.recall = not_built
            provider = make_provider(config=_LAYERED)
            _prefetch(provider)
        self.assertEqual(provider._consecutive_failures, 0)

    def test_the_memory_request_failing_counts_one_breaker_failure(self):
        with fake_backend() as fake:
            fake.errors["recall"] = RuntimeError("down")
            provider = make_provider(config=_LAYERED)
            _prefetch(provider)
        self.assertEqual(provider._consecutive_failures, 1)


class TestSlowServerProbe(unittest.TestCase):
    """After three prompts in a row whose recall only timed out, each prompt
    first spends a short /health probe and skips recall while the server does
    not answer (openclaw's slow-server probe)."""

    def _provider(self, fake, *, healthy):
        provider = make_provider(config=_LAYERED)
        fake.probe_calls = []

        def probe_health(**kwargs):
            fake.probe_calls.append(kwargs)
            return healthy

        fake.probe_health = probe_health
        return provider

    def test_three_timed_out_turns_arm_the_probe(self):
        with fake_backend() as fake:
            fake.errors["recall"] = TimeoutError("timed out")
            provider = self._provider(fake, healthy=False)
            for _ in range(provider_mod._SLOW_STREAK_THRESHOLD):
                _prefetch(provider)
            self.assertEqual(fake.probe_calls, [])
            before = len(fake.kwargs_for("recall"))
            _prefetch(provider)
            self.assertEqual(len(fake.probe_calls), 1)
            self.assertEqual(fake.probe_calls[0]["timeout"], provider_mod._READY_PROBE_TIMEOUT_SECS)
            # The server did not answer the probe, so no recall was sent.
            self.assertEqual(len(fake.kwargs_for("recall")), before)

    def test_a_healthy_probe_lets_recall_through_and_an_answer_ends_the_streak(self):
        with fake_backend() as fake:
            provider = self._provider(fake, healthy=True)
            provider._slow_streak = provider_mod._SLOW_STREAK_THRESHOLD
            fake.results["recall"] = [_OLD_SERVER_ITEM]
            out = _prefetch(provider)
        self.assertEqual(len(fake.probe_calls), 1)
        self.assertIn("<cognee_memory>", out)
        self.assertEqual(provider._slow_streak, 0)

    def test_other_failures_do_not_count_as_slow(self):
        with fake_backend() as fake:
            fake.errors["recall"] = RuntimeError("HTTP 500")
            provider = self._provider(fake, healthy=False)
            for _ in range(provider_mod._SLOW_STREAK_THRESHOLD + 1):
                _prefetch(provider)
        self.assertEqual(provider._slow_streak, 0)
        self.assertEqual(fake.probe_calls, [])


class TestSdkBackendLanes(unittest.TestCase):
    """The lanes through the in-process SDK transport.

    ``SdkBackend`` hands every call to one dedicated event loop and waits on
    ``future.result(timeout)``. Lanes submitted from pool threads must
    interleave as coroutines on that loop — not queue behind each other — and a
    lane that outlives its deadline must fail alone while the loop keeps
    serving the other.
    """

    def _sdk_provider(self, config=None, *, pinned=False):
        backend = SdkBackend()
        provider = make_provider(backend=backend, config={**_LAYERED, **(config or {})})
        provider._recall_pinned = pinned
        return backend, provider

    def test_memory_request_reaches_cognee_recall_explicitly(self):
        with fake_cognee() as fake:
            backend, provider = self._sdk_provider(pinned=True)
            try:
                _prefetch(provider)
                calls = fake.kwargs_for("recall")
            finally:
                backend.close(unregister=False)
        (graph,) = _by_scope(calls, "graph")
        self.assertEqual(graph["query_type"], "HYBRID_COMPLETION")
        self.assertIs(graph["only_context"], True)
        self.assertNotIn("session_id", graph)
        self.assertEqual(graph["datasets"], ["hermes"])
        (guidance,) = _by_scope(calls, "session_context")
        self.assertEqual(guidance["session_id"], "hermes_s-1")

    def test_an_unpinned_memory_request_names_no_dataset(self):
        with fake_cognee() as fake:
            backend, provider = self._sdk_provider()
            try:
                _prefetch(provider)
                calls = fake.kwargs_for("recall")
            finally:
                backend.close(unregister=False)
        (graph,) = _by_scope(calls, "graph")
        self.assertNotIn("datasets", graph)

    def test_lanes_interleave_on_the_single_sdk_loop(self):
        with fake_cognee() as fake:
            original = fake.recall

            async def slow(**kwargs):
                await asyncio.sleep(0.3)
                return await original(**kwargs)

            sys.modules["cognee"].recall = slow
            backend, provider = self._sdk_provider({"code_datasets": "codebase-svc-abc123"})
            try:
                started = time.monotonic()
                _prefetch(provider, _CODE_QUERY)
                wall = time.monotonic() - started
                calls = fake.kwargs_for("recall")
            finally:
                backend.close(unregister=False)
        self.assertEqual(len(calls), 3, calls)
        self.assertLess(wall, 0.55, f"lanes queued behind each other on the loop: {wall:.2f}s")

    def test_a_lane_past_its_deadline_fails_alone(self):
        with fake_cognee() as fake:
            original = fake.recall

            async def memory_hangs(**kwargs):
                # Only the memory lane names a search type; the others pass None.
                if kwargs.get("query_type") is not None:
                    await asyncio.sleep(3)
                await original(**kwargs)
                return [{"text": f"from {kwargs.get('scope')}"}]

            sys.modules["cognee"].recall = memory_hangs
            backend, provider = self._sdk_provider(
                {
                    "recall_timeout": 0.5,
                    "memory_hits": False,
                    "code_datasets": "codebase-svc-abc123",
                }
            )
            try:
                started = time.monotonic()
                out = _prefetch(provider, _CODE_QUERY)
                wall = time.monotonic() - started
            finally:
                backend.close(unregister=False)
        self.assertLess(wall, 1.5, f"the hung lane held the prefetch: {wall:.2f}s")
        self.assertIn("<code_graph>", out)
        self.assertNotIn("<cognee_memory>", out)
        # One lane timing out while the other answered is proof of life, not failure.
        self.assertEqual(provider._consecutive_failures, 0)


class TestCodeLane(unittest.TestCase):
    def test_configured_code_dataset_arms_the_lane_on_identifiers(self):
        with fake_backend() as fake:
            provider = make_provider(config=_WITH_CODE)
            provider.queue_prefetch(_CODE_QUERY)
            _settle(provider)
            (code_call,) = _by_scope(fake.kwargs_for("recall"), "code")
        self.assertEqual(code_call["datasets"], ["codebase-svc-abc123"])
        self.assertEqual(code_call["code_query"]["operation"], "query_facts")
        self.assertEqual(code_call["code_query"]["name"], "process_payment")

    def test_conversational_prompts_never_arm_the_code_lane(self):
        with fake_backend() as fake:
            provider = make_provider(config=_WITH_CODE)
            provider.queue_prefetch("how are you today")
            _settle(provider)
            calls = fake.kwargs_for("recall")
        self.assertEqual(_by_scope(calls, "code"), [])

    def test_code_graph_recall_off_disables_the_lane(self):
        with fake_backend() as fake:
            provider = make_provider(config={**_WITH_CODE, "code_graph_recall": False})
            provider.queue_prefetch(_CODE_QUERY)
            _settle(provider)
            calls = fake.kwargs_for("recall")
        self.assertEqual(_by_scope(calls, "code"), [])


class TestMemoryHitHeader(unittest.TestCase):
    def test_header_reports_hits_and_per_session_totals(self):
        with fake_backend() as fake:
            fake.results["recall"] = [_V160_ITEM]
            provider = make_provider(config={**_LAYERED, "memory_hits": True})
            out = _prefetch(provider, _QUERY)
        self.assertIn("1 memory hit this turn", out)
        self.assertNotIn("beyond this session", out)
        self.assertIn("1/1 turns had hits this session", out)

    def test_code_hits_count_toward_the_turn_without_a_provenance_note(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "remembered"}]
            provider = make_provider(config={**_WITH_CODE, "memory_hits": True})
            out = _prefetch(provider, _CODE_QUERY)
        self.assertIn("2 memory hits this turn", out)
        self.assertNotIn("beyond this session", out)

    def test_totals_accumulate_across_turns(self):
        with fake_backend() as fake:
            fake.results["recall"] = []
            provider = make_provider(config={**_LAYERED, "memory_hits": True})
            self.assertEqual(_prefetch(provider), "")  # turn 1: no hits
            fake.results["recall"] = [{"text": "remembered"}]
            out = _prefetch(provider)  # turn 2: hits
        self.assertIn("1/2 turns had hits this session", out)

    def test_reset_session_switch_clears_the_totals(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "remembered"}]
            provider = make_provider(config={**_LAYERED, "memory_hits": True})
            _prefetch(provider)
            provider.on_session_switch("s-2", reset=True)
        self.assertEqual(provider._turns_seen, 0)
        self.assertEqual(provider._hits_total, 0)

    def test_guidance_alone_gets_no_hit_header(self):
        with fake_backend() as fake:
            fake.results["recall:session_context"] = [{"text": "- a lesson"}]
            provider = make_provider(config={**_LAYERED, "memory_hits": True})
            out = _prefetch(provider)
        self.assertNotIn("memory hit", out)
        self.assertIn("<cognee_guidance>", out)

    def test_header_is_absent_when_disabled(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "remembered"}]
            provider = make_provider(config={**_LAYERED, "memory_hits": False})
            out = _prefetch(provider)
        self.assertNotIn("memory hit", out)


# The server scopes memory must never read: the raw session cache (cached
# turns, tool-call trace lessons) and the ``auto`` scope that folds it in.
# ``None`` is the same thing by omission — the server defaults to auto. The
# distilled ``session_context`` guidance is read on purpose, by its own lane.
_SESSION_SCOPES = ("session", "trace", "auto")


def _assert_explicit_scopes(case, calls):
    """Every recall stated its scope: the graph, the code graph, or guidance."""
    case.assertTrue(calls, "no recall was made")
    for kwargs in calls:
        scope = kwargs.get("scope")
        case.assertIn(scope, (["graph"], ["code"], ["session_context"]), kwargs)
        for retired in _SESSION_SCOPES:
            case.assertNotIn(retired, scope)
        if scope == ["graph"]:
            case.assertIsNone(kwargs.get("session_id"), kwargs)


class TestExplicitRecallReadsTheGraph(unittest.TestCase):
    """The ``cognee_recall`` tool targets the graph too; its old ``scope``
    argument (auto | session | graph) is gone, and a caller still passing one
    is ignored."""

    def _recall_kwargs(self, args):
        with fake_backend() as fake:
            provider = make_provider(session_cognee_id="hermes_abc")
            provider.handle_tool_call("cognee_recall", args)
            return fake.only_call("recall")

    def test_the_tool_sends_the_graph_scope_with_no_dataset_and_no_session(self):
        kwargs = self._recall_kwargs({"query": "q"})
        self.assertEqual(kwargs["scope"], ["graph"])
        # Auto recall (SDK-1393): unpinned, it names no dataset.
        self.assertIsNone(kwargs["datasets"])
        # As openclaw's memory_search: the conversation is already Hermes'.
        self.assertIsNone(kwargs["session_id"])
        self.assertIsNone(kwargs["query_type"])

    def test_a_legacy_scope_argument_never_changes_the_request(self):
        baseline = self._recall_kwargs({"query": "q"})
        for legacy in ("auto", "session", "graph", "GRAPH", "SESSION", None, ""):
            with self.subTest(scope=legacy):
                kwargs = self._recall_kwargs({"query": "q", "scope": legacy})
                self.assertEqual(kwargs, baseline)
                self.assertEqual(kwargs["scope"], ["graph"])

    def test_search_type_is_the_callers_override(self):
        kwargs = self._recall_kwargs({"query": "q", "scope": "session", "search_type": "CHUNKS"})
        self.assertEqual(kwargs["query_type"], "CHUNKS")
        self.assertEqual(kwargs["scope"], ["graph"])

    def test_the_tool_schema_has_no_scope_argument(self):
        with fake_backend():
            provider = make_provider()
            schema = next(s for s in provider.get_tool_schemas() if s["name"] == "cognee_recall")
        self.assertNotIn("scope", schema["parameters"]["properties"])
        for retired in ("session memory", "scope"):
            self.assertNotIn(retired, schema["description"].lower())


class TestNoCodePathReadsTheSessionCache(unittest.TestCase):
    """Every read path — the per-prompt prefetch with and without the code
    lane, the explicit tool under every legacy scope, and the code-search tool
    — states its scope. Nothing requests ``session``, ``trace`` or ``auto``,
    nothing leaves the scope to the server, and only the guidance lane sends a
    session id."""

    def test_every_read_path_states_its_scope(self):
        with fake_backend() as fake:
            provider = make_provider(config=_WITH_CODE)
            _prefetch(provider)
            _prefetch(provider, _CODE_QUERY)
            for legacy in ("auto", "session", "graph", None):
                provider.handle_tool_call("cognee_recall", {"query": "q", "scope": legacy})
            provider.handle_tool_call(
                "cognee_code_search",
                {"operation": "query_facts", "name": "process_payment", "dataset": "codebase-x"},
            )
            calls = fake.kwargs_for("recall")
        self.assertGreaterEqual(len(calls), 10, calls)
        _assert_explicit_scopes(self, calls)

    def test_the_sdk_transport_states_its_scope_too(self):
        with fake_cognee() as fake:
            backend = SdkBackend()
            provider = make_provider(backend=backend, config=_WITH_CODE)
            try:
                _prefetch(provider, _CODE_QUERY)
                provider.handle_tool_call("cognee_recall", {"query": "q", "scope": "session"})
                calls = fake.kwargs_for("recall")
            finally:
                backend.close(unregister=False)
        _assert_explicit_scopes(self, calls)

    def test_no_source_line_requests_a_session_cache_scope(self):
        # Belt and braces for the runtime tests above: the package must not even
        # spell a raw session-cache read. Prose uses backticks, not quoted
        # literals, so it does not match.
        import re

        forbidden = re.compile(
            r"""scope=["'](auto|session|trace)["']|\[["'](session|trace|auto)["']\]"""
            r"""|context_profile"""
        )
        package = ROOT / "cognee_integration_hermes"
        offenders = [
            f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
            for path in sorted(package.glob("*.py"))
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if forbidden.search(line)
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

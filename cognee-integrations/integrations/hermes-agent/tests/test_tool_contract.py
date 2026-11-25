"""Characterization: the model-visible tool contract and the request payloads.

Hermes hands ``handle_tool_call``'s return string straight to the model, so the
JSON envelope shapes here are a behavioural contract, not an implementation
detail. The recorded cognee kwargs are the other half of the contract: they are
what the HTTP request bodies must reproduce field for field.

Every test in this module must keep passing, unchanged, after the transport
swap. Run standalone with ``python3 tests/test_tool_contract.py``.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import (  # noqa: E402
    RememberResultStub,
    fake_backend,
    make_provider,
)


def _call(provider, tool, args):
    return json.loads(provider.handle_tool_call(tool, args))


class _ModelDumpItem:
    """A pydantic-style result object (``model_dump`` returning a dict)."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


# --------------------------------------------------------------------------
# Envelopes
# --------------------------------------------------------------------------


class TestRecallEnvelope(unittest.TestCase):
    def test_hit_returns_results_and_count(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "alpha"}, {"text": "beta"}]
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertEqual(out["count"], 2)
        self.assertEqual([item["text"] for item in out["results"]], ["alpha", "beta"])

    def test_miss_returns_result_message_and_zero_count(self):
        with fake_backend() as fake:
            fake.results["recall"] = []
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertEqual(out, {"result": "No relevant Cognee memory found.", "count": 0})

    def test_missing_query_is_an_error_and_calls_nothing(self):
        with fake_backend() as fake:
            provider = make_provider()
            out = _call(provider, "cognee_recall", {})
        self.assertEqual(out, {"error": "Missing required parameter: query"})
        self.assertEqual(fake.kwargs_for("recall"), [])

    def test_blank_query_is_an_error(self):
        with fake_backend():
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "   "})
        self.assertEqual(out, {"error": "Missing required parameter: query"})

    def test_backend_failure_is_reported_as_error_text(self):
        with fake_backend() as fake:
            fake.errors["recall"] = RuntimeError("boom")
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertTrue(out["error"].startswith("Cognee recall failed:"))
        self.assertIn("boom", out["error"])


class TestOverflowHintEnvelope(unittest.TestCase):
    """A backend that saw an embedding overflow gets it into the model's hands.

    The scan itself lives in ``test_overflow_hint.py``; here the contract is
    where the hint lands in the envelope: an ``error`` when recall came back
    empty, a ``warning`` when there are results — mean-pooled vectors match
    *something*, so plausible-but-wrong hits are the symptom worth flagging.
    """

    def test_non_empty_recall_carries_the_hint_as_a_warning(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "alpha"}]
            fake.overflow_hint_value = "index degrading"
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["warning"], "index degrading")

    def test_empty_recall_with_a_hint_is_a_hard_error(self):
        with fake_backend() as fake:
            fake.results["recall"] = []
            fake.overflow_hint_value = "index degrading"
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertEqual(out, {"error": "index degrading", "count": 0})

    def test_the_dim_mismatch_hint_wins_on_an_empty_recall(self):
        # A dimension mismatch means recall can never match — the more specific
        # diagnosis, so it outranks the overflow hint.
        with fake_backend() as fake:
            fake.results["recall"] = []
            fake.empty_recall_hint_value = "dim mismatch"
            fake.overflow_hint_value = "index degrading"
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertEqual(out["error"], "dim mismatch")

    def test_no_warning_key_without_a_hint(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"text": "alpha"}]
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertNotIn("warning", out)

    def test_remember_success_carries_the_hint_as_a_warning(self):
        with fake_backend() as fake:
            fake.overflow_hint_value = "index degrading"
            provider = make_provider()
            out = _call(provider, "cognee_remember", {"content": "fact"})
        self.assertEqual(out["result"], "Content stored in Cognee.")
        self.assertEqual(out["warning"], "index degrading")

    def test_remember_success_has_no_warning_without_a_hint(self):
        with fake_backend():
            provider = make_provider()
            out = _call(provider, "cognee_remember", {"content": "fact"})
        self.assertNotIn("warning", out)


class TestRecallTimeoutAdvice(unittest.TestCase):
    """A timed-out recall tells the model the fast way out, inline in the error."""

    def test_a_timeout_error_names_the_faster_search_type(self):
        with fake_backend() as fake:
            fake.errors["recall"] = TimeoutError()
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertIn("CHUNKS", out["error"])
        self.assertIn("COGNEE_RECALL_TIMEOUT", out["error"])

    def test_a_timed_out_message_gets_the_advice_too(self):
        # The HTTP transport wraps urllib timeouts in CogneeUnreachable, whose
        # type says nothing — the message is the only timeout signal.
        with fake_backend() as fake:
            fake.errors["recall"] = RuntimeError("cognee unreachable at http://x: timed out")
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertIn("CHUNKS", out["error"])

    def test_a_non_timeout_failure_stays_unadorned(self):
        with fake_backend() as fake:
            fake.errors["recall"] = RuntimeError("boom")
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        self.assertNotIn("CHUNKS", out["error"])


class TestRememberEnvelope(unittest.TestCase):
    def test_success_envelope(self):
        with fake_backend() as fake:
            fake.results["remember_permanent"] = RememberResultStub("running")
            provider = make_provider()
            out = _call(provider, "cognee_remember", {"content": "fact"})
        self.assertEqual(out, {"result": "Content stored in Cognee.", "status": "running"})

    def test_status_falls_back_to_completed_without_a_status_attribute(self):
        # NOTE for the HTTP swap: the SDK returns a RememberResult *object*, so
        # ``getattr(result, "status", ...)`` picks up the real status. An HTTP
        # backend returns a *dict*, which has no ``.status`` attribute — so it
        # would silently always report "completed". The HTTP backend must either
        # return a status-bearing object or the handler must read dict keys too.
        with fake_backend() as fake:
            fake.results["remember_permanent"] = {"status": "running"}
            provider = make_provider()
            out = _call(provider, "cognee_remember", {"content": "fact"})
        self.assertEqual(out["status"], "completed")

    def test_missing_content_is_an_error_and_calls_nothing(self):
        with fake_backend() as fake:
            provider = make_provider()
            out = _call(provider, "cognee_remember", {})
        self.assertEqual(out, {"error": "Missing required parameter: content"})
        self.assertEqual(fake.kwargs_for("remember_permanent"), [])

    def test_backend_failure_is_reported_as_error_text(self):
        with fake_backend() as fake:
            fake.errors["remember_permanent"] = RuntimeError("nope")
            provider = make_provider()
            out = _call(provider, "cognee_remember", {"content": "fact"})
        self.assertTrue(out["error"].startswith("Cognee remember failed:"))


_DATASET_ID = "11111111-1111-1111-1111-111111111111"
_DATA_ID_A = "22222222-2222-2222-2222-222222222222"
_DATA_ID_B = "33333333-3333-3333-3333-333333333333"


class TestForgetEnvelope(unittest.TestCase):
    """The two-phase forget: find lists candidates, forget deletes confirmed ids."""

    def test_missing_action_is_an_error_and_touches_nothing(self):
        with fake_backend() as fake:
            provider = make_provider()
            out = _call(provider, "cognee_forget", {})
        self.assertIn("action='find'", out["error"])
        self.assertEqual(fake.calls, [])

    def test_find_requires_terms(self):
        with fake_backend() as fake:
            provider = make_provider()
            out = _call(provider, "cognee_forget", {"action": "find"})
        self.assertIn("terms", out["error"])
        self.assertEqual(fake.calls, [])

    def test_find_lists_matching_documents_with_previews(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = [{"id": _DATASET_ID, "name": "hermes"}]
            fake.results["list_dataset_data"] = [
                {"id": _DATA_ID_A, "name": "doc-a"},
                {"id": _DATA_ID_B, "name": "doc-b"},
            ]
            raw = {
                _DATA_ID_A: "we talked about tennis and rackets",
                _DATA_ID_B: "grocery list: milk, eggs",
            }
            original = fake.read_raw_data

            def read_raw_data(**kwargs):
                original(**kwargs)
                return raw[kwargs["data_id"]]

            fake.read_raw_data = read_raw_data
            provider = make_provider()
            out = _call(provider, "cognee_forget", {"action": "find", "terms": "tennis"})
        self.assertEqual(out["action"], "find")
        self.assertEqual(len(out["candidates"]), 1)
        candidate = out["candidates"][0]
        self.assertEqual(candidate["data_id"], _DATA_ID_A)
        self.assertEqual(candidate["matched_terms"], ["tennis"])
        self.assertIn("tennis", candidate["preview"])
        # Nothing was deleted during a find.
        self.assertEqual(fake.kwargs_for("forget_document"), [])
        self.assertEqual(fake.kwargs_for("forget"), [])

    def test_find_with_no_matches_says_so(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = [{"id": _DATASET_ID, "name": "hermes"}]
            fake.results["list_dataset_data"] = []
            provider = make_provider()
            out = _call(provider, "cognee_forget", {"action": "find", "terms": "tennis"})
        self.assertEqual(out["candidates"], [])
        self.assertIn("No stored documents matched", out["result"])

    def test_forget_requires_confirm(self):
        with fake_backend() as fake:
            provider = make_provider()
            out = _call(provider, "cognee_forget", {"action": "forget", "data_ids": [_DATA_ID_A]})
        self.assertIn("confirm=true", out["error"])
        self.assertEqual(fake.calls, [])

    def test_forget_requires_data_ids(self):
        with fake_backend() as fake:
            provider = make_provider()
            out = _call(provider, "cognee_forget", {"action": "forget", "confirm": True})
        self.assertIn("data_ids", out["error"])
        self.assertEqual(fake.kwargs_for("forget_document"), [])

    def test_forget_deletes_exactly_the_listed_ids(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = [{"id": _DATASET_ID, "name": "hermes"}]
            provider = make_provider()
            out = _call(
                provider,
                "cognee_forget",
                {"action": "forget", "data_ids": [_DATA_ID_A, _DATA_ID_B], "confirm": True},
            )
        self.assertEqual(out["deleted"], [_DATA_ID_A, _DATA_ID_B])
        deletes = fake.kwargs_for("forget_document")
        self.assertEqual([d["data_id"] for d in deletes], [_DATA_ID_A, _DATA_ID_B])
        self.assertTrue(all(d["dataset_id"] == _DATASET_ID for d in deletes))

    def test_a_failed_delete_is_reported_per_id_not_thrown(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = [{"id": _DATASET_ID, "name": "hermes"}]
            fake.errors["forget_document"] = RuntimeError("denied")
            provider = make_provider()
            out = _call(
                provider,
                "cognee_forget",
                {"action": "forget", "data_ids": [_DATA_ID_A], "confirm": True},
            )
        self.assertEqual(out["deleted"], [])
        self.assertEqual(out["errors"][0]["data_id"], _DATA_ID_A)

    def test_everything_in_dataset_uses_the_coarse_endpoint(self):
        with fake_backend() as fake:
            fake.results["forget"] = {"deleted": 3}
            provider = make_provider()
            out = _call(
                provider,
                "cognee_forget",
                {"action": "forget", "everything_in_dataset": True, "confirm": True},
            )
            kwargs = fake.only_call("forget")
        self.assertIn("deleted", out["result"].lower() + str(out))
        self.assertEqual(kwargs["dataset"], "hermes")
        # The all-datasets wipe is not expressible from the tool, by construction.
        self.assertIs(kwargs["everything"], False)


class TestDispatch(unittest.TestCase):
    def test_unknown_tool_names_the_tool(self):
        with fake_backend():
            provider = make_provider()
            out = _call(provider, "cognee_nope", {})
        self.assertEqual(out, {"error": "Unknown Cognee tool: cognee_nope"})

    def test_open_breaker_short_circuits_every_tool_without_calling_the_backend(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider._is_breaker_open = lambda: True
            for tool, args in (
                ("cognee_recall", {"query": "q"}),
                ("cognee_remember", {"content": "c"}),
                ("cognee_forget", {"everything": True}),
            ):
                out = _call(provider, tool, args)
                self.assertIn("temporarily unavailable", out["error"])
        self.assertEqual(fake.calls, [])

    def test_tool_schemas_are_the_six_declared_tools_with_required_params(self):
        provider = make_provider()
        schemas = {schema["name"]: schema for schema in provider.get_tool_schemas()}
        self.assertEqual(
            set(schemas),
            {
                "cognee_recall",
                "cognee_remember",
                "cognee_forget",
                "cognee_switch_dataset",
                "cognee_code_search",
                "cognee_recap",
            },
        )
        self.assertEqual(schemas["cognee_recall"]["parameters"]["required"], ["query"])
        self.assertEqual(schemas["cognee_remember"]["parameters"]["required"], ["content"])
        self.assertEqual(schemas["cognee_forget"]["parameters"]["required"], ["action"])
        self.assertEqual(schemas["cognee_switch_dataset"]["parameters"]["required"], ["action"])
        self.assertEqual(schemas["cognee_code_search"]["parameters"]["required"], ["operation"])
        self.assertEqual(schemas["cognee_recap"]["parameters"]["required"], ["kind"])
        self.assertEqual(
            set(schemas["cognee_recall"]["parameters"]["properties"]),
            {"query", "search_type", "top_k"},
        )

    def test_optional_tools_can_be_disabled_by_config(self):
        provider = make_provider(
            config={"dataset_switch_tool": False, "code_search_tool": False, "recap_tool": False}
        )
        names = {schema["name"] for schema in provider.get_tool_schemas()}
        self.assertEqual(names, {"cognee_recall", "cognee_remember", "cognee_forget"})


# --------------------------------------------------------------------------
# Result normalization — decides what text the model actually sees
# --------------------------------------------------------------------------


class TestResultNormalization(unittest.TestCase):
    def _first(self, item):
        with fake_backend() as fake:
            fake.results["recall"] = [item]
            provider = make_provider()
            out = _call(provider, "cognee_recall", {"query": "q"})
        return out["results"][0]

    def test_answer_wins_over_text_and_content(self):
        item = self._first({"answer": "A", "text": "T", "content": "C", "summary": "S"})
        self.assertEqual(item["text"], "A")

    def test_text_wins_over_content(self):
        self.assertEqual(self._first({"text": "T", "content": "C"})["text"], "T")

    def test_content_wins_over_chunk_text(self):
        self.assertEqual(self._first({"content": "C", "chunk_text": "K"})["text"], "C")

    def test_chunk_text_wins_over_summary(self):
        self.assertEqual(self._first({"chunk_text": "K", "summary": "S"})["text"], "K")

    def test_summary_is_the_last_resort_key(self):
        self.assertEqual(self._first({"summary": "S"})["text"], "S")

    def test_model_dump_objects_are_coerced(self):
        item = self._first(_ModelDumpItem({"answer": "from-model-dump"}))
        self.assertEqual(item["text"], "from-model-dump")

    def test_plain_string_results_become_text(self):
        self.assertEqual(self._first("bare string")["text"], "bare string")

    def test_source_defaults_to_cognee(self):
        self.assertEqual(self._first({"text": "T"})["source"], "cognee")

    def test_source_prefers_source_then_underscore_source(self):
        self.assertEqual(self._first({"text": "T", "source": "graph"})["source"], "graph")
        self.assertEqual(self._first({"text": "T", "_source": "session"})["source"], "session")

    def test_optional_metadata_included_only_when_present(self):
        rich = self._first(
            {
                "text": "T",
                "score": 0.5,
                "dataset": "d",
                "dataset_name": "dn",
                "node_name": "nn",
            }
        )
        self.assertEqual(rich["score"], 0.5)
        self.assertEqual(rich["dataset"], "d")
        self.assertEqual(rich["dataset_name"], "dn")
        self.assertEqual(rich["node_name"], "nn")

        plain = self._first({"text": "T", "score": None})
        self.assertEqual(set(plain), {"text", "source"})


# --------------------------------------------------------------------------
# Request payloads — what the HTTP bodies must reproduce
# --------------------------------------------------------------------------


class TestRecallPayload(unittest.TestCase):
    def _recall_kwargs(self, args, **provider_kwargs):
        with fake_backend() as fake:
            provider = make_provider(**provider_kwargs)
            provider.handle_tool_call("cognee_recall", args)
            return fake.only_call("recall")

    def test_query_top_k_and_auto_route_always_sent(self):
        kwargs = self._recall_kwargs({"query": "hello"}, top_k=7, auto_route=False)
        self.assertEqual(kwargs["query"], "hello")
        self.assertEqual(kwargs["top_k"], 7)
        self.assertIs(kwargs["auto_route"], False)

    def test_recall_targets_the_graph_with_no_dataset_and_no_session(self):
        kwargs = self._recall_kwargs({"query": "q"}, session_cognee_id="hermes_abc")
        self.assertEqual(kwargs["scope"], ["graph"])
        # Auto recall (SDK-1393): unpinned, it names no dataset and the server picks.
        self.assertIsNone(kwargs["datasets"])
        # As openclaw's memory_search: the conversation is already Hermes'.
        self.assertIsNone(kwargs["session_id"])

    def test_a_legacy_scope_argument_is_ignored(self):
        # The tool no longer takes ``scope``; an older caller passing one gets
        # the same graph request as everyone else.
        for legacy in (None, "auto", "session", "graph", "GRAPH"):
            with self.subTest(scope=legacy):
                kwargs = self._recall_kwargs({"query": "q", "scope": legacy})
                self.assertEqual(kwargs["scope"], ["graph"])
                self.assertIsNone(kwargs["datasets"])
                self.assertIsNone(kwargs["session_id"])

    def test_search_type_is_applied(self):
        kwargs = self._recall_kwargs({"query": "q", "search_type": "CHUNKS"})
        self.assertEqual(kwargs["query_type"], "CHUNKS")

    def test_unrecognized_search_type_is_passed_through(self):
        # The provider does not police search-type names; each transport resolves
        # them and falls back (see test_sdk_backend). That keeps a search type
        # added by a future cognee working without a provider change.
        kwargs = self._recall_kwargs({"query": "q", "search_type": "NOT_A_TYPE"})
        self.assertEqual(kwargs["query_type"], "NOT_A_TYPE")

    def test_no_query_type_when_search_type_absent(self):
        self.assertIsNone(self._recall_kwargs({"query": "q"})["query_type"])


class TestTopKClamping(unittest.TestCase):
    def _top_k(self, args, configured=5):
        with fake_backend() as fake:
            provider = make_provider(top_k=configured)
            provider.handle_tool_call("cognee_recall", {"query": "q", **args})
            return fake.only_call("recall")["top_k"]

    def test_absent_top_k_uses_provider_default(self):
        self.assertEqual(self._top_k({}, configured=9), 9)

    def test_zero_is_falsy_and_falls_back_to_provider_default(self):
        self.assertEqual(self._top_k({"top_k": 0}, configured=9), 9)

    def test_negative_clamps_to_one(self):
        self.assertEqual(self._top_k({"top_k": -3}), 1)

    def test_above_twenty_clamps_to_twenty(self):
        self.assertEqual(self._top_k({"top_k": 50}), 20)

    def test_in_range_passes_through(self):
        self.assertEqual(self._top_k({"top_k": 12}), 12)

    def test_numeric_string_is_accepted(self):
        self.assertEqual(self._top_k({"top_k": "8"}), 8)


class TestRememberPayload(unittest.TestCase):
    def test_explicit_remember_targets_the_permanent_graph(self):
        with fake_backend() as fake:
            provider = make_provider(session_cognee_id="hermes_s9")
            provider.handle_tool_call("cognee_remember", {"content": "durable fact"})
            kwargs = fake.only_call("remember_permanent")
        self.assertEqual(kwargs["text"], "durable fact")
        self.assertEqual(kwargs["dataset"], "hermes")
        # Linked to the current session so the graph write keeps its provenance.
        self.assertEqual(kwargs["session_ids"], ["hermes_s9"])

    def test_explicit_remember_does_not_use_the_session_cache(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.handle_tool_call("cognee_remember", {"content": "durable fact"})
            self.assertEqual(fake.kwargs_for("remember_session"), [])

    def test_dataset_argument_overrides_the_provider_dataset(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.handle_tool_call("cognee_remember", {"content": "c", "dataset": "other"})
            self.assertEqual(fake.only_call("remember_permanent")["dataset"], "other")

    def test_content_is_stripped(self):
        with fake_backend() as fake:
            provider = make_provider()
            provider.handle_tool_call("cognee_remember", {"content": "  padded  "})
            self.assertEqual(fake.only_call("remember_permanent")["text"], "padded")


class TestForgetPayload(unittest.TestCase):
    def test_document_delete_resolves_the_dataset_id_by_name(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = [
                {"id": _DATASET_ID, "name": "hermes"},
                {"id": _DATA_ID_B, "name": "other"},
            ]
            provider = make_provider()
            provider.handle_tool_call(
                "cognee_forget",
                {"action": "forget", "data_ids": [_DATA_ID_A], "confirm": True},
            )
            kwargs = fake.only_call("forget_document")
        self.assertEqual(kwargs["dataset_id"], _DATASET_ID)
        self.assertEqual(kwargs["data_id"], _DATA_ID_A)

    def test_an_explicit_dataset_argument_wins_over_the_provider_default(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = [{"id": _DATASET_ID, "name": "special"}]
            provider = make_provider()
            provider.handle_tool_call(
                "cognee_forget",
                {
                    "action": "forget",
                    "dataset": "special",
                    "data_ids": [_DATA_ID_A],
                    "confirm": True,
                },
            )
            self.assertEqual(fake.only_call("forget_document")["dataset_id"], _DATASET_ID)

    def test_an_unknown_dataset_is_an_error_before_any_delete(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = []
            provider = make_provider()
            out = json.loads(
                provider.handle_tool_call(
                    "cognee_forget",
                    {"action": "forget", "data_ids": [_DATA_ID_A], "confirm": True},
                )
            )
        self.assertIn("not found", out["error"])
        self.assertEqual(fake.kwargs_for("forget_document"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

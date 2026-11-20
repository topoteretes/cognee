"""Wire-level tests for ``SdkBackend``: protocol args in, cognee kwargs out.

The provider tests assert on protocol args (``test_tool_contract``,
``test_lifecycle_contract``); this file asserts that the SDK transport faithfully
turns those into what cognee expects. That is the whole split: Hermes semantics
above the seam, wire format below it. An HTTP transport gets an equivalent file.

Two things only exist below the seam and are pinned here: the ``user=`` kwarg
rule, and the SDK-only fields (``self_improvement``, and dropping ``dataset`` on
a wipe-everything).

Runs standalone with ``python3 tests/test_sdk_backend.py``; needs no real cognee.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import fake_cognee  # noqa: E402
from cognee_integration_hermes.backend import SdkBackend  # noqa: E402

_TIMEOUT = 5.0


def _backend(*, served=False, user=None):
    backend = SdkBackend()
    backend.served = served
    backend._user = user
    # Run coroutines inline instead of on the bridge's loop thread.
    backend._bridge = _InlineBridge()
    return backend


class _InlineBridge:
    def run(self, coro, timeout=None):
        import asyncio

        return asyncio.run(coro)

    def shutdown(self):
        pass


class TestRecallWireFormat(unittest.TestCase):
    def _recall_kwargs(self, **overrides):
        params = {
            "query": "q",
            "session_id": "hermes_s1",
            "datasets": ["hermes"],
            "top_k": 5,
            "auto_route": True,
            "query_type": None,
            "timeout": _TIMEOUT,
        }
        params.update(overrides)
        with fake_cognee() as fake:
            _backend(served=True).recall(**params)
            return fake.only_call("recall")

    def test_query_becomes_query_text(self):
        self.assertEqual(self._recall_kwargs(query="hello")["query_text"], "hello")

    def test_top_k_and_auto_route_pass_through(self):
        kwargs = self._recall_kwargs(top_k=9, auto_route=False)
        self.assertEqual(kwargs["top_k"], 9)
        self.assertIs(kwargs["auto_route"], False)

    def test_none_targets_are_omitted_not_sent_as_none(self):
        # cognee distinguishes an omitted kwarg from an explicit None, so a
        # session-only or graph-only recall must leave the other key out.
        session_only = self._recall_kwargs(datasets=None)
        self.assertIn("session_id", session_only)
        self.assertNotIn("datasets", session_only)

        graph_only = self._recall_kwargs(session_id=None)
        self.assertIn("datasets", graph_only)
        self.assertNotIn("session_id", graph_only)

    def test_query_type_is_resolved_to_a_search_type(self):
        self.assertEqual(self._recall_kwargs(query_type="CHUNKS")["query_type"], "CHUNKS")

    def test_query_type_is_uppercased(self):
        self.assertEqual(self._recall_kwargs(query_type="chunks")["query_type"], "CHUNKS")

    def test_unknown_query_type_falls_back_to_graph_completion(self):
        kwargs = self._recall_kwargs(query_type="NOT_A_TYPE")
        self.assertEqual(kwargs["query_type"], "GRAPH_COMPLETION")

    def test_absent_query_type_is_omitted(self):
        self.assertNotIn("query_type", self._recall_kwargs(query_type=None))

    def test_memory_lane_request_is_forwarded_explicitly(self):
        # The per-prompt memory lane: scope and only_context travel to
        # cognee.recall alongside the resolved SearchType and the session id.
        kwargs = self._recall_kwargs(
            query_type="HYBRID_COMPLETION", scope=["graph"], only_context=True
        )
        self.assertEqual(kwargs["query_type"], "HYBRID_COMPLETION")
        self.assertEqual(kwargs["scope"], ["graph"])
        self.assertIs(kwargs["only_context"], True)
        self.assertEqual(kwargs["session_id"], "hermes_s1")

    def test_absent_scope_is_stated_as_the_graph(self):
        # Left out, cognee.recall resolves the scope to ``auto`` and folds the
        # session cache in alongside the session id; memory reads the graph only.
        kwargs = self._recall_kwargs()
        self.assertEqual(kwargs["scope"], ["graph"])
        self.assertNotIn("only_context", kwargs)

    def test_http_only_fields_are_not_forwarded(self):
        kwargs = self._recall_kwargs(scope=["code"], code_query={"operation": "query_facts"})
        self.assertNotIn("code_query", kwargs)
        self.assertEqual(kwargs["scope"], ["code"])


class TestRememberWireFormat(unittest.TestCase):
    def test_session_write_is_a_cheap_cache_write(self):
        with fake_cognee() as fake:
            _backend(served=True).remember_session(
                text="turn", session_id="hermes_s1", dataset="hermes", timeout=_TIMEOUT
            )
            kwargs = fake.only_call("remember")
        self.assertEqual(kwargs["data"], "turn")
        self.assertEqual(kwargs["dataset_name"], "hermes")
        self.assertEqual(kwargs["session_id"], "hermes_s1")
        self.assertIs(kwargs["self_improvement"], False)
        self.assertNotIn("session_ids", kwargs)

    def test_permanent_write_requests_graph_extraction(self):
        with fake_cognee() as fake:
            _backend(served=True).remember_permanent(
                text="fact", dataset="hermes", session_ids=["hermes_s1"], timeout=_TIMEOUT
            )
            kwargs = fake.only_call("remember")
        self.assertEqual(kwargs["data"], "fact")
        self.assertEqual(kwargs["dataset_name"], "hermes")
        self.assertIs(kwargs["self_improvement"], True)
        self.assertEqual(kwargs["session_ids"], ["hermes_s1"])
        self.assertNotIn("session_id", kwargs)


class TestForgetWireFormat(unittest.TestCase):
    def _forget_kwargs(self, **overrides):
        params = {
            "dataset": "hermes",
            "everything": False,
            "memory_only": False,
            "timeout": _TIMEOUT,
        }
        params.update(overrides)
        with fake_cognee() as fake:
            _backend(served=True).forget(**params)
            return fake.only_call("forget")

    def test_dataset_scoped_delete(self):
        kwargs = self._forget_kwargs()
        self.assertEqual(kwargs["dataset"], "hermes")
        self.assertIs(kwargs["everything"], False)

    def test_everything_drops_the_dataset(self):
        # Sending both would scope a wipe-everything to one dataset.
        kwargs = self._forget_kwargs(everything=True)
        self.assertIs(kwargs["everything"], True)
        self.assertNotIn("dataset", kwargs)

    def test_no_dataset_is_omitted(self):
        self.assertNotIn("dataset", self._forget_kwargs(dataset=None))

    def test_memory_only_passes_through(self):
        self.assertIs(self._forget_kwargs(memory_only=True)["memory_only"], True)


class TestImproveWireFormat(unittest.TestCase):
    def test_background_becomes_run_in_background(self):
        with fake_cognee() as fake:
            _backend(served=True).improve(
                dataset="hermes",
                session_ids=["hermes_s1"],
                background=True,
                timeout=_TIMEOUT,
            )
            kwargs = fake.only_call("improve")
        self.assertEqual(kwargs["dataset"], "hermes")
        self.assertEqual(kwargs["session_ids"], ["hermes_s1"])
        self.assertIs(kwargs["run_in_background"], True)

    def test_foreground_is_sent_explicitly(self):
        with fake_cognee() as fake:
            _backend(served=True).improve(
                dataset="hermes", session_ids=[], background=False, timeout=_TIMEOUT
            )
            self.assertIs(fake.only_call("improve")["run_in_background"], False)


class TestUserKwarg(unittest.TestCase):
    """Identity is the transport's business, and only in-process.

    A served instance owns identity via the api-key principal. Passing
    ``user=None`` is not the same as omitting it — the SDK may treat an explicit
    None differently — so it must be left out entirely.
    """

    def test_omitted_when_served_even_with_a_user_set(self):
        backend = _backend(served=True, user="USER")
        kwargs = {}
        backend._add_user_kwarg(kwargs)
        self.assertNotIn("user", kwargs)

    def test_included_when_in_process(self):
        backend = _backend(served=False, user="USER")
        kwargs = {}
        backend._add_user_kwarg(kwargs)
        self.assertEqual(kwargs["user"], "USER")

    def test_omitted_when_there_is_no_user(self):
        backend = _backend(served=False, user=None)
        kwargs = {}
        backend._add_user_kwarg(kwargs)
        self.assertNotIn("user", kwargs)

    def test_reaches_the_operations_in_process(self):
        with fake_cognee() as fake:
            _backend(served=False, user="USER").remember_permanent(
                text="fact", dataset="hermes", session_ids=[], timeout=_TIMEOUT
            )
            self.assertEqual(fake.only_call("remember")["user"], "USER")


class TestConnectAndClose(unittest.TestCase):
    def test_connect_serves_the_url_and_marks_served(self):
        with fake_cognee() as fake:
            backend = _backend()
            backend.connect(url="http://127.0.0.1:8000", api_key="", timeout=_TIMEOUT)
            self.assertTrue(backend.served)
            self.assertEqual(fake.only_call("serve"), {"url": "http://127.0.0.1:8000"})

    def test_api_key_is_only_sent_when_present(self):
        with fake_cognee() as fake:
            _backend().connect(url="https://cloud.example", api_key="ck_1", timeout=_TIMEOUT)
            self.assertEqual(fake.only_call("serve")["api_key"], "ck_1")

    def test_close_disconnects_only_when_served(self):
        with fake_cognee() as fake:
            _backend(served=True).close(timeout=_TIMEOUT)
            self.assertEqual(len(fake.kwargs_for("disconnect")), 1)

        with fake_cognee() as fake:
            _backend(served=False).close(timeout=_TIMEOUT)
            self.assertEqual(fake.kwargs_for("disconnect"), [])


class TestEmptyRecallHint(unittest.TestCase):
    def test_skipped_when_served(self):
        # A served instance owns the vector store; the local engine says nothing.
        self.assertIsNone(_backend(served=True).empty_recall_hint())


if __name__ == "__main__":
    unittest.main(verbosity=2)

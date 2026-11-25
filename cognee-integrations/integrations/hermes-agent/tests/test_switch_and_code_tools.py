"""The cognee_switch_dataset and cognee_code_search tool contracts.

Run standalone with ``python3 tests/test_switch_and_code_tools.py``.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes import code_graph, dataset_overrides  # noqa: E402


def _call(provider, tool, args):
    return json.loads(provider.handle_tool_call(tool, args))


class TestSwitchDataset(unittest.TestCase):
    def test_current_reports_the_active_and_default_dataset(self):
        with fake_backend():
            provider = make_provider()
            out = _call(provider, "cognee_switch_dataset", {"action": "current"})
        self.assertEqual(out["dataset"], "hermes")
        self.assertEqual(out["default"], "hermes")
        self.assertFalse(out["switched"])

    def test_list_marks_the_current_dataset(self):
        with fake_backend() as fake:
            fake.results["list_datasets"] = [
                {"id": "x", "name": "hermes"},
                {"id": "y", "name": "work"},
            ]
            provider = make_provider()
            out = _call(provider, "cognee_switch_dataset", {"action": "list"})
        rows = {row["name"]: row["current"] for row in out["datasets"]}
        self.assertEqual(rows, {"hermes": True, "work": False})

    def test_switch_bridges_ensures_and_repoints(self):
        with fake_backend() as fake:
            provider = make_provider()
            out = _call(provider, "cognee_switch_dataset", {"action": "switch", "dataset": "work"})
        self.assertTrue(out["switched"])
        self.assertEqual(out["dataset"], "work")
        self.assertTrue(out["previous"]["bridged"])
        # 1. The session being left was bridged into its OLD dataset, on the
        #    server's own background pipeline.
        improve = fake.only_call("improve")
        self.assertEqual(improve["dataset"], "hermes")
        self.assertEqual(improve["session_ids"], ["hermes_s-1"])
        self.assertTrue(improve["background"])
        # 2. The target was ensured.
        self.assertEqual(fake.only_call("ensure_dataset")["dataset"], "work")
        # 3. Capture and recall now target the new dataset under a fresh cognee
        #    session id (a cognee session never spans two datasets).
        self.assertEqual(provider._dataset, "work")
        self.assertEqual(provider._session_cognee_id, "hermes_s-1__1")
        # 4. The override is persisted for a later resume of this conversation.
        override = dataset_overrides.load_override("s-1")
        self.assertEqual(override["dataset"], "work")
        self.assertEqual(override["counter"], 1)

    def test_switch_to_the_active_dataset_only_restricts_recall(self):
        """SDK-1393: picking the dataset in use pins recall to it, with no bridge,
        no new session and no backend call; picking it again is a no-op."""
        with fake_backend() as fake:
            provider = make_provider()
            out = _call(
                provider, "cognee_switch_dataset", {"action": "switch", "dataset": "hermes"}
            )
            again = _call(
                provider, "cognee_switch_dataset", {"action": "switch", "dataset": "hermes"}
            )
        self.assertFalse(out["switched"])
        self.assertEqual(out["reason"], "recall_restricted")
        self.assertEqual(again["reason"], "already_active")
        self.assertTrue(provider._recall_pinned)
        self.assertEqual(provider._session_cognee_id, "hermes_s-1")
        self.assertEqual(fake.calls, [])
        # No switch: the mode alone is stored, so it applies to the active dataset.
        override = dataset_overrides.load_override("s-1")
        self.assertEqual(override["recall"], "pinned")
        self.assertNotIn("dataset", override)
        self.assertNotIn("counter", override)

    def test_a_failed_bridge_aborts_without_force(self):
        with fake_backend() as fake:
            fake.errors["improve"] = RuntimeError("server busy")
            provider = make_provider()
            out = _call(provider, "cognee_switch_dataset", {"action": "switch", "dataset": "work"})
        self.assertIn("force=true", out["error"])
        # Nothing changed: same dataset, same session id, target never ensured.
        self.assertEqual(provider._dataset, "hermes")
        self.assertEqual(provider._session_cognee_id, "hermes_s-1")
        self.assertEqual(fake.kwargs_for("ensure_dataset"), [])

    def test_force_defers_the_failed_bridge_to_session_end(self):
        with fake_backend() as fake:
            fake.errors["improve"] = RuntimeError("server busy")
            provider = make_provider()
            out = _call(
                provider,
                "cognee_switch_dataset",
                {"action": "switch", "dataset": "work", "force": True},
            )
            self.assertTrue(out["switched"])
            self.assertFalse(out["previous"]["bridged"])
            self.assertEqual(provider._retired_sessions, [("hermes", "hermes_s-1")])
            # At session end the retired session is re-submitted (and the
            # current one closed as usual).
            del fake.errors["improve"]
            provider.on_session_end([])
            improves = fake.kwargs_for("improve")
        self.assertEqual(improves[0]["dataset"], "hermes")
        self.assertEqual(improves[0]["session_ids"], ["hermes_s-1"])
        self.assertEqual(provider._retired_sessions, [])

    def test_an_unavailable_target_dataset_is_an_error(self):
        with fake_backend() as fake:
            fake.errors["ensure_dataset"] = RuntimeError("403")
            provider = make_provider()
            out = _call(provider, "cognee_switch_dataset", {"action": "switch", "dataset": "work"})
        self.assertIn("not available", out["error"])
        self.assertEqual(provider._dataset, "hermes")

    def test_reset_returns_to_the_default_dataset(self):
        with fake_backend():
            provider = make_provider()
            _call(provider, "cognee_switch_dataset", {"action": "switch", "dataset": "work"})
            out = _call(provider, "cognee_switch_dataset", {"action": "reset"})
        self.assertTrue(out["switched"])
        self.assertEqual(out["dataset"], "hermes")
        # Every switch mints a fresh session id — the counter never reuses one.
        self.assertEqual(provider._session_cognee_id, "hermes_s-1__2")

    def test_a_persisted_override_is_reapplied_on_initialize(self):
        dataset_overrides.save_override("sess-9", "work", 3)
        with fake_backend():
            provider = make_provider(session_id="sess-9")
            provider._session_cognee_id = "hermes_sess-9"
            provider._apply_dataset_override()
        self.assertEqual(provider._dataset, "work")
        self.assertEqual(provider._session_cognee_id, "hermes_sess-9__3")
        self.assertEqual(provider._switch_counter, 3)


class TestCodeSearch(unittest.TestCase):
    def _provider(self, **kwargs):
        provider = make_provider(**kwargs)
        provider._cwd = ""  # no ambient repo unless a test sets one
        return provider

    def test_unknown_operation_is_an_error(self):
        with fake_backend():
            out = _call(self._provider(), "cognee_code_search", {"operation": "drop_tables"})
        self.assertIn("Unknown operation", out["error"])

    def test_without_an_indexed_repo_the_error_names_the_cli(self):
        with fake_backend():
            out = _call(
                self._provider(),
                "cognee_code_search",
                {"operation": "query_facts", "name": "foo"},
            )
        self.assertIn("hermes cognee index-repo", out["error"])

    def test_query_facts_builds_the_bounded_code_query(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"operation": "query_facts", "facts": []}]
            provider = self._provider(config={"code_datasets": "codebase-svc-abc"})
            out = _call(
                provider,
                "cognee_code_search",
                {"operation": "query_facts", "name": "process_payment", "limit": 7},
            )
            kwargs = fake.only_call("recall")
        self.assertEqual(out["dataset"], "codebase-svc-abc")
        self.assertEqual(kwargs["scope"], ["code"])
        self.assertEqual(kwargs["datasets"], ["codebase-svc-abc"])
        self.assertEqual(
            kwargs["code_query"],
            {"operation": "query_facts", "name": "process_payment", "limit": 7},
        )

    def test_operation_specific_shapes(self):
        cases = [
            ({"operation": "explore", "name": "A"}, {"operation": "explore", "name": "A"}),
            ({"operation": "traverse", "name": "A"}, {"operation": "traverse", "start": "A"}),
            (
                {"operation": "find_path", "name": "A", "target": "B"},
                {"operation": "find_path", "source": "A", "target": "B"},
            ),
            (
                {"operation": "impact_analysis", "name": "A"},
                {"operation": "impact_analysis", "targets": ["A"]},
            ),
            ({"operation": "delta"}, {"operation": "delta"}),
        ]
        for args, expected in cases:
            with fake_backend() as fake:
                provider = self._provider(config={"code_datasets": "codebase-svc-abc"})
                _call(provider, "cognee_code_search", args)
                self.assertEqual(
                    fake.only_call("recall")["code_query"], expected, args["operation"]
                )

    def test_seed_requiring_operations_demand_a_name(self):
        for operation in ("explore", "traverse", "impact_analysis"):
            with fake_backend() as fake:
                provider = self._provider(config={"code_datasets": "codebase-svc-abc"})
                out = _call(provider, "cognee_code_search", {"operation": operation})
                self.assertIn("requires name", out["error"], operation)
                self.assertEqual(fake.kwargs_for("recall"), [])

    def test_repo_argument_resolves_via_the_registry(self):
        with fake_backend() as fake:
            code_graph.record_index(
                "https://github.com/a/svc", "codebase-svc-def", index_vectors=False
            )
            provider = self._provider()
            out = _call(
                provider,
                "cognee_code_search",
                {"operation": "delta", "repo": "https://github.com/a/svc"},
            )
        self.assertEqual(out["dataset"], "codebase-svc-def")
        self.assertEqual(fake.only_call("recall")["datasets"], ["codebase-svc-def"])

    def test_results_pass_through_as_structured_dicts(self):
        with fake_backend() as fake:
            fake.results["recall"] = [{"operation": "delta", "added": 3, "removed": 1}]
            provider = self._provider(config={"code_datasets": "codebase-svc-abc"})
            out = _call(provider, "cognee_code_search", {"operation": "delta"})
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["results"][0]["added"], 3)


class TestAutoRecall(unittest.TestCase):
    """SDK-1393: graph recall names no dataset unless the conversation pinned
    it (or the configured recall_mode is "pinned"): the server picks."""

    def _graph_recall(self, provider, fake):
        _call(provider, "cognee_recall", {"query": "what did we decide?"})
        (kwargs,) = [k for k in fake.kwargs_for("recall") if k["scope"] == ["graph"]]
        return kwargs

    def test_a_switch_pins_recall_and_auto_lifts_it(self):
        with fake_backend() as fake:
            provider = make_provider()
            _call(provider, "cognee_switch_dataset", {"action": "switch", "dataset": "work"})
            self.assertEqual(self._graph_recall(provider, fake)["datasets"], ["work"])
            out = _call(provider, "cognee_switch_dataset", {"action": "auto"})
            self.assertEqual(out["reason"], "recall_auto")
            fake.calls.clear()
            self.assertIsNone(self._graph_recall(provider, fake)["datasets"])
            # Saves stay in the switched dataset, and a resume keeps both the
            # switch and the mode, whatever the config now says.
            self.assertEqual(provider._dataset, "work")
            override = dataset_overrides.load_override("s-1")
            self.assertEqual(
                (override["dataset"], override["counter"], override["recall"]), ("work", 1, "auto")
            )
            resumed = make_provider(dataset="research")
            resumed._configured_pins_recall = True
            resumed._apply_dataset_override()
            self.assertEqual(resumed._dataset, "work")
            fake.calls.clear()
            self.assertIsNone(self._graph_recall(resumed, fake)["datasets"])

    def test_switching_to_the_default_dataset_pins(self):
        with fake_backend():
            provider = make_provider()
            _call(provider, "cognee_switch_dataset", {"action": "switch", "dataset": "work"})
            _call(provider, "cognee_switch_dataset", {"action": "auto"})
            out = _call(
                provider, "cognee_switch_dataset", {"action": "switch", "dataset": "hermes"}
            )
            self.assertTrue(out["switched"])
            self.assertEqual(out["recall"], "pinned")
            current = _call(provider, "cognee_switch_dataset", {"action": "current"})
        self.assertEqual((current["dataset"], current["recall"]), ("hermes", "pinned"))
        self.assertEqual(dataset_overrides.load_override("s-1")["recall"], "pinned")

    def test_auto_is_not_a_dataset_name(self):
        for name in ("auto", " AUTO "):
            with fake_backend() as fake:
                provider = make_provider()
                out = _call(
                    provider, "cognee_switch_dataset", {"action": "switch", "dataset": name}
                )
            self.assertIn("action='auto'", out["error"], name)
            self.assertEqual(fake.calls, [], name)
            self.assertEqual(provider._dataset, "hermes")
            self.assertFalse(provider._recall_pinned)


if __name__ == "__main__":
    unittest.main(verbosity=2)

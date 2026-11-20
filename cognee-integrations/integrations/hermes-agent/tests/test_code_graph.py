"""The code-graph helpers: identifier gate, repo identity, state registry.

Run standalone with ``python3 tests/test_code_graph.py``.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import code_graph  # noqa: E402


class TestExtractIdentifiers(unittest.TestCase):
    """The syntactic gate that keeps the code lane off conversational prompts."""

    def test_conversational_prompts_yield_nothing(self):
        for prompt in (
            "how are you today",
            "summarize what we discussed yesterday",
            "what's the weather like",
            "",
        ):
            self.assertEqual(code_graph.extract_identifiers(prompt), [], prompt)

    def test_snake_case_fires(self):
        self.assertEqual(
            code_graph.extract_identifiers("what calls process_payment?"),
            ["process_payment"],
        )

    def test_camel_case_fires(self):
        self.assertIn("UserService", code_graph.extract_identifiers("explain UserService"))

    def test_file_paths_fire(self):
        self.assertIn(
            "billing/api.py", code_graph.extract_identifiers("look at billing/api.py please")
        )

    def test_backticked_identifiers_win(self):
        found = code_graph.extract_identifiers("check `resolve_config` and process_payment")
        self.assertEqual(found[0], "resolve_config")

    def test_backticked_prose_is_ignored(self):
        self.assertEqual(code_graph.extract_identifiers("run `git status now` for me"), [])

    def test_stoplist_words_do_not_fire(self):
        self.assertEqual(code_graph.extract_identifiers("ask Cognee about GitHub"), [])
        self.assertEqual(code_graph.extract_identifiers("Hermes and OpenAI teamed up"), [])

    def test_domains_and_prose_dots_do_not_fire(self):
        self.assertEqual(code_graph.extract_identifiers("see example.com e.g. today"), [])

    def test_at_most_limit_tokens(self):
        found = code_graph.extract_identifiers("a_b c_d e_f g_h", limit=2)
        self.assertEqual(len(found), 2)


class TestRepoIdentity(unittest.TestCase):
    def test_remote_specs_are_canonicalized(self):
        self.assertEqual(
            code_graph.canonical_spec("https://github.com/a/repo.git"),
            "https://github.com/a/repo",
        )
        self.assertEqual(
            code_graph.canonical_spec("https://github.com/a/repo/"),
            "https://github.com/a/repo",
        )

    def test_dataset_name_is_stable_and_prefixed(self):
        name = code_graph.default_code_dataset("https://github.com/a/repo")
        self.assertTrue(name.startswith("codebase-repo-"))
        self.assertEqual(name, code_graph.default_code_dataset("https://github.com/a/repo.git"))

    def test_same_basename_different_paths_get_different_datasets(self):
        # The path digest is load-bearing: two checkouts sharing a basename in
        # one dataset would let each re-index's stale-node sweep delete the
        # other's nodes.
        a = code_graph.default_code_dataset("https://github.com/a/service")
        b = code_graph.default_code_dataset("https://github.com/b/service")
        self.assertNotEqual(a, b)


class TestRepoStateRegistry(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        patcher = mock.patch.object(code_graph, "_STATE_DIR", self.tmp / "state")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_record_and_find_by_cwd(self):
        repo = self.tmp / "repo"
        (repo / "src").mkdir(parents=True)
        state = code_graph.record_index(str(repo), "codebase-repo-x", index_vectors=False)
        self.assertEqual(state["spec_kind"], "path")
        found = code_graph.find_indexed_repo(str(repo / "src"))
        self.assertEqual(found["dataset"], "codebase-repo-x")
        # Outside the repo: no match.
        self.assertEqual(code_graph.find_indexed_repo(str(self.tmp)), {})

    def test_nested_checkouts_resolve_to_the_innermost_repo(self):
        outer = self.tmp / "outer"
        inner = outer / "vendor" / "inner"
        inner.mkdir(parents=True)
        code_graph.record_index(str(outer), "codebase-outer-1", index_vectors=False)
        code_graph.record_index(str(inner), "codebase-inner-1", index_vectors=False)
        self.assertEqual(code_graph.find_indexed_repo(str(inner))["dataset"], "codebase-inner-1")
        self.assertEqual(code_graph.find_indexed_repo(str(outer))["dataset"], "codebase-outer-1")

    def test_url_indexed_repos_have_no_local_root(self):
        state = code_graph.record_index(
            "https://github.com/a/repo", "codebase-repo-y", index_vectors=True
        )
        self.assertEqual(state["spec_kind"], "url")
        self.assertEqual(state["repo_root"], "")
        self.assertEqual(code_graph.find_indexed_repo(str(self.tmp)), {})
        # But it is findable by spec and by dataset name.
        self.assertEqual(
            code_graph.find_repo_state("https://github.com/a/repo.git")["dataset"],
            "codebase-repo-y",
        )
        self.assertEqual(
            code_graph.find_repo_state("codebase-repo-y")["dataset"], "codebase-repo-y"
        )

    def test_auto_code_lane_needs_both_gates(self):
        repo = self.tmp / "repo"
        repo.mkdir()
        # Identifier without an indexed repo: no lane.
        self.assertEqual(code_graph.auto_code_lane("what calls foo_bar", str(repo)), {})
        code_graph.record_index(str(repo), "codebase-repo-z", index_vectors=False)
        # Indexed repo without an identifier: no lane.
        self.assertEqual(code_graph.auto_code_lane("how are you", str(repo)), {})
        # Both: the lane fires with a bounded query_facts seed.
        lane = code_graph.auto_code_lane("what calls foo_bar", str(repo))
        self.assertEqual(lane["dataset"], "codebase-repo-z")
        self.assertEqual(lane["identifier"], "foo_bar")
        self.assertEqual(lane["code_query"]["operation"], "query_facts")

    def test_invalid_state_files_are_skipped(self):
        state_dir = self.tmp / "state"
        state_dir.mkdir(parents=True)
        (state_dir / "broken.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(code_graph.load_repo_states(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

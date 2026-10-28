"""Real Enola -> Ladybug -> answer context, over a small repository written by the test."""

import importlib
from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.code_completion_retriever import CodeCompletionRetriever
from cognee.modules.retrieval.code_retriever import invalidate_code_graph_snapshot_cache
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.tasks.code_graph.enola import EnolaNotInstalledError, find_enola_binary

SYNC = """def _sync_repositories(credential):
    return credential


def sync_repositories(credential, trigger="manual"):
    if trigger != "manual":
        return "paused"
    return _sync_repositories(credential)


def unrelated():
    return 1
"""

RETIRE = """def assert_tracked_selection(db):
    return db


def _forget_repository_urls(db, urls):
    for url in urls:
        forget(data_id=url)


def record_indexed_repository(credential):
    return credential


def retire_unselected_repositories(db, urls):
    _forget_repository_urls(db, urls)
"""

VERIFY = """import hmac


def verify(signature, expected):
    return hmac.compare_digest(signature, expected)
"""


@pytest.mark.asyncio
async def test_questions_have_actual_source_evidence(tmp_path, monkeypatch):
    try:
        find_enola_binary()
    except EnolaNotInstalledError:
        pytest.skip("Enola binary is required for the extraction integration test")
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    extractor = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")
    graph_module = importlib.import_module("cognee.infrastructure.databases.graph.get_graph_engine")
    code_module = importlib.import_module("cognee.modules.retrieval.code_retriever")
    root = tmp_path / "github"
    root.mkdir()
    (root / "sync.py").write_text(SYNC)
    (root / "retire.py").write_text(RETIRE)
    (root / "verify_signature.py").write_text(VERIFY)
    engine = LadybugAdapter(db_path=str(tmp_path / "graph.lbug"))
    monkeypatch.setattr(graph_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(code_module, "get_graph_engine", AsyncMock(return_value=engine))
    answer_module = importlib.import_module("cognee.modules.retrieval.code_completion_retriever")
    monkeypatch.setattr(answer_module, "get_graph_engine", AsyncMock(return_value=engine))
    invalidate_code_graph_snapshot_cache(all_entries=True)
    try:
        points = await extractor.extract_code_graph(repo_path=root)
        state = await extractor.add_code_graph_data_points(points)
        await extractor.add_code_graph_edges(state, repo_path=root)
        text = GraphCompletionRetriever()
        text.get_retrieved_objects = AsyncMock(side_effect=NoDataError("No searchable text"))
        # Exact questions use explicit operations here: this test verifies the
        # extractor/database/evidence contract without paying for a model.
        for question, terms, expected in (
            ("Da li pauza blokira ručni sync?", "sync_repositories", 'trigger != "manual"'),
            (
                "Šta uklanja izbačen repozitorijum?",
                "retire_unselected_repositories",
                "_forget_repository_urls",
            ),
            ("Gde se proverava webhook potpis?", "verify", "compare_digest"),
        ):
            retriever = CodeCompletionRetriever(text, {"operation": "ask", "question": terms})
            objects = await retriever.get_retrieved_objects(question)
            sources = objects["code_evidence"]["sources"]
            assert any(expected in (s["source"] or {}).get("text", "") for s in sources), question
            for source in sources:
                if source["source"]:
                    original = (root / source["file"]).read_text().splitlines(keepends=True)
                    evidence = source["source"]
                    assert evidence["text"] == "".join(
                        original[evidence["line"] - 1 : evidence["end_line"]]
                    )
        # The evidence of a symbol is that symbol, not its file: a function followed by others
        # does not carry them.
        forget_urls = await CodeCompletionRetriever(
            text, {"operation": "ask", "question": "_forget_repository_urls"}
        ).get_retrieved_objects("forget")
        evidence = next(
            s["source"]
            for s in forget_urls["code_evidence"]["sources"]
            if s["name"].endswith("._forget_repository_urls") and s["source"]
        )
        assert "forget(data_id=" in evidence["text"]
        assert "def record_indexed_repository" not in evidence["text"]
        assert "def assert_tracked_selection" not in evidence["text"]
        assert "def retire_unselected_repositories" not in evidence["text"]
        for operation in ("neighbors", "impact_analysis"):
            config = {"operation": operation, "name": "_sync_repositories"}
            if operation == "neighbors":
                config.update(direction="reverse", relation_types=["calls"])
            objects = await CodeCompletionRetriever(text, config).get_retrieved_objects(
                "Ko poziva ovu funkciju i šta njena promena pogađa?"
            )
            assert any(
                s["name"].endswith(".sync_repositories")
                for s in objects["code_evidence"]["sources"]
            )
        missing = await CodeCompletionRetriever(text, {"operation": "ask"}).get_retrieved_objects(
            "zzzz_nonexistent_provider_zzzz"
        )
        assert missing["code_evidence"]["sources"] == []
        assert missing["code_evidence"]["operations"][0]["coverage"]["notes"]
        # Stored evidence survives a later checkout edit; search never reads it.
        (root / "sync.py").write_text("# unrelated later revision\n")
        result = await CodeCompletionRetriever(text, {"operation": "ask"}).get_retrieved_objects(
            "sync_repositories"
        )
        assert any(
            'trigger != "manual"' in (s["source"] or {}).get("text", "")
            for s in result["code_evidence"]["sources"]
        )
    finally:
        invalidate_code_graph_snapshot_cache(all_entries=True)
        await engine.close()

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.recall.methods.normalize_search_payload import normalize_search_payload
from cognee.modules.retrieval.code_completion_retriever import (
    CodeCompletionRetriever,
    CodeQuestionPlan,
)
from cognee.modules.retrieval.code_retriever import CodeRetriever, _CodeGraphSnapshot
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType
from cognee.tasks.code_graph.source_evidence import source_node_id

module = importlib.import_module("cognee.modules.retrieval.code_completion_retriever")


@pytest.fixture(autouse=True)
def preferences(monkeypatch):
    monkeypatch.setattr(module, "load_preference_text", AsyncMock(return_value=""))


SYMBOL_SOURCE = {
    "text": "def verify_signature(signature):\n    return compare_digest(signature, expected)\n",
    "line": 1,
    "end_line": 2,
    "text_sha256": "hash-a",
}


@pytest.fixture
def stored_nodes():
    """The symbols as a CODE query loads them: a hash of the source, never its text."""
    return [
        (
            "verify",
            {
                "type": "CodeSymbol",
                "kind": "symbol",
                "name": "verify_signature",
                "file_path": "webhook.py",
                "line": 1,
                "end_line": 2,
                "repo": "acme/app",
                "doc": "Verify webhook signature before processing the event.",
                "source_sha256": "hash-a",
            },
        ),
        (
            "handler",
            {
                "type": "CodeSymbol",
                "kind": "symbol",
                "name": "handle_webhook",
                "file_path": "webhook.py",
                "line": 5,
                "repo": "acme/app",
            },
        ),
    ]


@pytest.fixture
def source_nodes():
    """The CodeSource nodes a CODE query never loads, by node id."""
    return {str(source_node_id("verify")): dict(SYMBOL_SOURCE)}


@pytest.fixture
def snapshot(monkeypatch, stored_nodes, source_nodes):
    graph = _CodeGraphSnapshot(stored_nodes, [("handler", "verify", "calls", {})])
    monkeypatch.setattr(CodeRetriever, "_snapshot", AsyncMock(return_value=graph))

    async def get_nodes(ids):
        return [
            {"id": node_id, **source_nodes[node_id]} for node_id in ids if node_id in source_nodes
        ]

    engine = SimpleNamespace(get_nodes=AsyncMock(side_effect=get_nodes))
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=engine))
    graph.engine = engine
    return graph


@pytest.fixture
def text():
    retriever = GraphCompletionRetriever()
    retriever.get_retrieved_objects = AsyncMock(return_value=[])
    retriever.get_context_from_objects = AsyncMock(
        return_value="Document: webhook requests are signed."
    )
    return retriever


@pytest.mark.asyncio
async def test_serbian_question_retrieves_english_symbol_and_source(snapshot, text, monkeypatch):
    planner = AsyncMock(
        return_value=CodeQuestionPlan(terms="verify webhook signature", intent="explain")
    )
    monkeypatch.setattr(module, "plan_code_question", planner)
    retriever = CodeCompletionRetriever(text, None)
    objects = await retriever.get_retrieved_objects("Gde se proverava webhook potpis?")
    context = json.loads(await retriever.get_context_from_objects(retrieved_objects=objects))
    assert context["documents"].startswith("Document:")
    evidence = context["code"]
    source = next(s for s in evidence["sources"] if s["node_id"] == "verify")
    assert "compare_digest" in source["source"]["text"]
    assert source["source"]["text_sha256"] == "hash-a"
    assert "handler" in evidence["missing_source_ids"]
    assert "compare_digest" not in retriever.system_prompt
    planner.assert_awaited_once_with("Gde se proverava webhook potpis?")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent,operation", [("callers", "neighbors"), ("impact", "impact_analysis")]
)
async def test_graph_questions_use_retrieved_ids(snapshot, text, monkeypatch, intent, operation):
    monkeypatch.setattr(
        module,
        "plan_code_question",
        AsyncMock(return_value=CodeQuestionPlan(terms="verify_signature", intent=intent)),
    )
    result = await CodeCompletionRetriever(text, None).get_retrieved_objects(
        "Ko zavisi od provere potpisa?"
    )
    operations = result["code_evidence"]["operations"]
    assert operations[1]["operation"] == operation
    assert "handler" in json.dumps(operations[1])


@pytest.mark.asyncio
async def test_code_only_dataset_keeps_evidence(snapshot, text):
    text.get_retrieved_objects.side_effect = NoDataError("No searchable text")
    result = await CodeCompletionRetriever(text, {"operation": "ask"}).get_retrieved_objects(
        "verify_signature"
    )
    assert result["text_context"] == ""
    assert result["code_evidence"]["sources"]
    assert "No searchable text" in result["code_evidence"]["text_status"]


@pytest.mark.asyncio
async def test_code_failure_is_not_replaced_by_text_answer(snapshot, text, monkeypatch):
    monkeypatch.setattr(
        CodeRetriever, "_snapshot", AsyncMock(side_effect=RuntimeError("graph unavailable"))
    )
    with pytest.raises(RuntimeError, match="graph unavailable"):
        await CodeCompletionRetriever(text, None).get_retrieved_objects("question")
    text.get_retrieved_objects.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_infrastructure_failure_propagates(snapshot, text):
    text.get_retrieved_objects.side_effect = RuntimeError("connection lost")
    with pytest.raises(RuntimeError, match="connection lost"):
        await CodeCompletionRetriever(text, {"operation": "ask"}).get_retrieved_objects(
            "verify_signature"
        )


@pytest.mark.asyncio
async def test_no_match_keeps_coverage_note(snapshot, text):
    result = await CodeCompletionRetriever(text, {"operation": "ask"}).get_retrieved_objects(
        "zzzzzzzz"
    )
    assert result["code_evidence"]["sources"] == []
    assert "No symbol" in result["code_evidence"]["operations"][0]["coverage"]["notes"][0]


@pytest.mark.asyncio
async def test_only_context_does_not_plan_with_llm(snapshot, text, monkeypatch):
    planner = AsyncMock(side_effect=AssertionError("preview called LLM"))
    monkeypatch.setattr(module, "plan_code_question", planner)
    await CodeCompletionRetriever(text, None, plan_question=False).get_retrieved_objects(
        "verify_signature"
    )
    planner.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_source_is_not_cut_at_arbitrary_character_limit(
    snapshot, text, source_nodes
):
    source_nodes[str(source_node_id("verify"))]["text"] = "# evidence\n" * 3000
    retriever = CodeCompletionRetriever(text, {"operation": "ask"})
    objects = await retriever.get_retrieved_objects("verify_signature")
    context = json.loads(await retriever.get_context_from_objects(retrieved_objects=objects))
    source = next(s for s in context["code"]["sources"] if s["node_id"] == "verify")
    assert source["source"]["text"].count("# evidence") == 3000


def test_answer_normalization_preserves_code_evidence():
    evidence = {"sources": [{"file": "webhook.py", "line": 1}], "missing_source_ids": []}
    payload = SearchResultPayload(
        search_type=SearchType.GRAPH_COMPLETION, completion="Answer", code_evidence=evidence
    )
    result = normalize_search_payload(payload)[0]
    assert result.metadata["code_evidence"] == evidence


@pytest.mark.asyncio
async def test_model_receives_evidence_as_context(snapshot, text, monkeypatch):
    answer = AsyncMock(return_value="Potpis se proverava u webhook.py:1.")
    monkeypatch.setattr(module, "generate_answer", answer)
    retriever = CodeCompletionRetriever(text, {"operation": "ask"})
    monkeypatch.setattr(retriever, "_use_session_cache", lambda: False)
    objects = await retriever.get_retrieved_objects("verify_signature")
    context = await retriever.get_context_from_objects(retrieved_objects=objects)
    result = await retriever.get_completion_from_context(
        query="Gde se proverava potpis?", context=context, retrieved_objects=objects
    )
    assert result == ["Potpis se proverava u webhook.py:1."]
    assert "compare_digest" in answer.await_args.kwargs["context"]
    assert "compare_digest" not in answer.await_args.kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_factory_and_search_payload_use_shared_completion_pipeline(
    snapshot, text, monkeypatch
):
    factory = importlib.import_module(
        "cognee.modules.search.methods.get_search_type_retriever_instance"
    )
    output = importlib.import_module("cognee.modules.search.methods.get_retriever_output")
    graph = SimpleNamespace(is_empty=AsyncMock(return_value=False))
    monkeypatch.setattr(output, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(
        GraphCompletionRetriever,
        "get_retrieved_objects",
        AsyncMock(side_effect=NoDataError("no text")),
    )
    monkeypatch.setattr(
        CodeCompletionRetriever,
        "prepare_session_turn_for_retrieval",
        AsyncMock(
            return_value=SimpleNamespace(should_answer=True, effective_query="verify_signature")
        ),
    )
    monkeypatch.setattr(CodeCompletionRetriever, "_use_session_cache", lambda self: False)
    monkeypatch.setattr(
        module, "generate_answer", AsyncMock(return_value="Verified in webhook.py:1")
    )
    result = await output.get_retriever_output(
        SearchType.GRAPH_COMPLETION,
        "verify_signature",
        retriever_specific_config={"code_context": {"operation": "ask"}},
    )
    assert result.completion == ["Verified in webhook.py:1"]
    assert any(s["node_id"] == "verify" for s in result.code_evidence["sources"])
    preview = await factory.get_search_type_retriever_instance(
        SearchType.GRAPH_COMPLETION,
        "verify_signature",
        only_context=True,
        retriever_specific_config={"code_context": None},
    )
    assert not preview.plan_question


@pytest.mark.asyncio
async def test_concurrent_session_does_not_hide_code_failure(text, monkeypatch):
    session = importlib.import_module("cognee.modules.retrieval.session_aware_completion")
    monkeypatch.setattr(session, "warm_preference_cache", AsyncMock())
    retriever = CodeCompletionRetriever(text, None)
    retriever.get_retrieved_objects = AsyncMock(
        side_effect=[RuntimeError("code lookup failed"), {"code_evidence": {}}]
    )
    context = SimpleNamespace(recent_qas=(("qa1", "earlier question", "earlier answer"),))
    with pytest.raises(RuntimeError, match="code lookup failed"):
        await session._retrieve_merged_objects(
            retriever, raw_query="current question", snapshot=context
        )


@pytest.mark.asyncio
async def test_session_completion_receives_same_evidence_once(snapshot, text, monkeypatch):
    graph_module = importlib.import_module("cognee.modules.retrieval.graph_completion_retriever")
    manager = SimpleNamespace(generate_completion_with_session=AsyncMock(return_value="Answer"))
    monkeypatch.setattr(graph_module, "get_session_manager", lambda: manager)
    retriever = CodeCompletionRetriever(text, {"operation": "ask"})
    retriever.session_id = "code-conversation"
    monkeypatch.setattr(retriever, "_use_session_cache", lambda: True)
    objects = await retriever.get_retrieved_objects("verify_signature")
    context = await retriever.get_context_from_objects(retrieved_objects=objects)
    await retriever.get_completion_from_context(
        query="question", context=context, retrieved_objects=objects
    )
    manager.generate_completion_with_session.assert_awaited_once()
    assert manager.generate_completion_with_session.await_args.kwargs["context"] == context
    assert (
        manager.generate_completion_with_session.await_args.kwargs["session_id"]
        == "code-conversation"
    )


@pytest.mark.asyncio
async def test_a_question_is_planned_once_for_every_dataset_and_lane(snapshot, text, monkeypatch):
    plan = AsyncMock(return_value=CodeQuestionPlan(terms="verify_signature", intent="explain"))
    monkeypatch.setattr(module, "plan_code_question", plan)
    cache = {}
    retrievers = [CodeCompletionRetriever(text, None, plan_cache=cache) for _ in range(3)]
    await asyncio.gather(
        *(r.get_retrieved_objects("Da li se proverava potpis?") for r in retrievers)
    )
    assert plan.await_count == 1
    await retrievers[0].get_retrieved_objects("Another question")
    assert plan.await_count == 2


def test_text_evidence_helpers_tolerate_a_turn_without_text(text):
    retriever = CodeCompletionRetriever(text, {"operation": "ask"})
    for objects in (None, {"text_objects": None, "code_evidence": {}}):
        assert retriever.get_context_evidence(objects) == []
        assert retriever.extract_context_object_ids(objects) is None


def test_a_strict_retriever_is_declared_on_the_base_class():
    from cognee.modules.retrieval.base_retriever import BaseRetriever

    assert BaseRetriever.strict_retrieval is False
    assert CodeCompletionRetriever.strict_retrieval is True


def _source(node_id, line, end_line, kind="CodeSymbol", text="body", file="a.py"):
    return {
        "node_id": node_id,
        "type": kind,
        "name": node_id,
        "repo": "r",
        "file": file,
        "line": line,
        "end_line": end_line,
        "source": {"text": text, "line": line, "end_line": end_line, "text_sha256": "h"}
        if text
        else None,
    }


def test_a_text_inside_another_source_is_named_not_repeated():
    sources = [
        _source("class", 1, 20, text="class"),
        _source("method", 5, 9, text="method"),
        _source("other_file", 5, 9, file="b.py", text="elsewhere"),
    ]
    result = {
        s["node_id"]: s["source"] for s in CodeCompletionRetriever._without_nested_text(sources)
    }
    assert result["class"]["text"] == "class"
    assert "text" not in result["method"] and result["method"]["within"] == "class"
    assert result["other_file"]["text"] == "elsewhere"


def test_the_outermost_source_keeps_the_text_of_a_nesting_chain():
    sources = [
        _source("outer", 1, 30, text="outer"),
        _source("middle", 5, 20, text="middle"),
        _source("inner", 6, 8, text="inner"),
    ]
    result = {
        s["node_id"]: s["source"] for s in CodeCompletionRetriever._without_nested_text(sources)
    }
    assert result["middle"]["within"] == result["inner"]["within"] == "outer"


def test_only_symbols_are_reported_as_missing_source():
    sources = [
        _source("symbol", 1, 2, text=None),
        _source("insight", 0, 0, kind="CodeInsight", text=None),
        _source("module", 0, 0, kind="CodeModule", text=None),
        _source("covered", 3, 4),
    ]
    assert CodeCompletionRetriever._missing(sources) == ["symbol"]


@pytest.mark.asyncio
async def test_merged_lanes_drop_a_no_text_status_once_any_lane_found_text(snapshot, text):
    retriever = CodeCompletionRetriever(text, {"operation": "ask"})
    first = await retriever.get_retrieved_objects("verify_signature")
    first["code_evidence"]["text_status"] = "No searchable text"
    first["text_context"] = ""
    second = await retriever.get_retrieved_objects("verify_signature")
    merged = retriever.merge_retrieved_objects(first, second)
    assert merged["text_context"] and "text_status" not in merged["code_evidence"]
    both_empty = retriever.merge_retrieved_objects(first, {**first})
    assert both_empty["code_evidence"]["text_status"] == "No searchable text"


@pytest.mark.asyncio
async def test_a_seed_the_graph_cannot_resolve_is_a_coverage_note_not_a_failure(snapshot, text):
    retriever = CodeCompletionRetriever(
        text, {"operation": "explore", "name": "no_such_symbol_anywhere"}
    )
    objects = await retriever.get_retrieved_objects("question")
    assert objects["code_evidence"]["sources"] == []
    assert objects["code_evidence"]["operations"][0]["coverage"]["notes"]


@pytest.mark.asyncio
async def test_one_cancelled_asker_does_not_cancel_the_shared_plan(snapshot, text, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(question):
        started.set()
        await release.wait()
        return CodeQuestionPlan(terms="verify_signature", intent="explain")

    monkeypatch.setattr(module, "plan_code_question", slow)
    cache = {}
    first = asyncio.create_task(CodeCompletionRetriever(text, None, plan_cache=cache)._plan("q"))
    await started.wait()
    second = asyncio.create_task(CodeCompletionRetriever(text, None, plan_cache=cache)._plan("q"))
    await asyncio.sleep(0)
    first.cancel()
    release.set()
    assert (await second).terms == "verify_signature"


@pytest.mark.asyncio
async def test_a_failed_conversational_lane_fails_a_strict_turn(text, monkeypatch):
    session = importlib.import_module("cognee.modules.retrieval.session_aware_completion")
    monkeypatch.setattr(session, "warm_preference_cache", AsyncMock())
    retriever = CodeCompletionRetriever(text, None)
    retriever.get_retrieved_objects = AsyncMock(
        side_effect=[{"code_evidence": {}}, RuntimeError("rewrite lane failed")]
    )
    context = SimpleNamespace(recent_qas=(("qa1", "earlier question", "earlier answer"),))
    with pytest.raises(RuntimeError, match="rewrite lane failed"):
        await session._retrieve_merged_objects(
            retriever, raw_query="current question", snapshot=context
        )


@pytest.mark.asyncio
async def test_the_factory_refuses_code_context_for_a_type_that_cannot_carry_it():
    from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
    from cognee.modules.search.methods.get_search_type_retriever_instance import (
        get_search_type_retriever_instance,
    )

    with pytest.raises(QueryValidationError, match="Code answers support"):
        await get_search_type_retriever_instance(
            SearchType.CHUNKS,
            "question",
            retriever_specific_config={"code_context": None},
        )


def test_the_snapshot_holds_no_source_text_only_which_symbols_have_some(snapshot):
    assert snapshot.source_hashes == {"verify": "hash-a"}
    assert all("source_sha256" not in node for node in snapshot.nodes.values())


@pytest.mark.asyncio
async def test_source_text_is_read_by_id_for_the_cited_symbols_only(snapshot, text):
    await CodeCompletionRetriever(text, {"operation": "ask"}).get_retrieved_objects(
        "verify_signature"
    )
    asked = {id_ for call in snapshot.engine.get_nodes.await_args_list for id_ in call.args[0]}
    # Only the source node of the symbol that has stored source is read.
    assert asked == {str(source_node_id("verify"))}


@pytest.mark.asyncio
async def test_a_cited_symbol_without_stored_source_is_reported_missing(snapshot, text):
    objects = await CodeCompletionRetriever(
        text, {"operation": "ask", "question": "handle_webhook"}
    ).get_retrieved_objects("handle_webhook")
    assert "handler" in objects["code_evidence"]["missing_source_ids"]
    assert "verify" not in objects["code_evidence"]["missing_source_ids"]


@pytest.mark.asyncio
async def test_text_that_was_re_indexed_since_the_snapshot_is_stale_not_cited(
    snapshot, text, source_nodes
):
    source_nodes[str(source_node_id("verify"))]["text_sha256"] = "hash-b"
    objects = await CodeCompletionRetriever(text, {"operation": "ask"}).get_retrieved_objects(
        "verify_signature"
    )
    evidence = objects["code_evidence"]
    assert evidence["stale_source_ids"] == ["verify"]
    assert "verify" not in evidence["missing_source_ids"]
    assert not next(s for s in evidence["sources"] if s["node_id"] == "verify")["source"]


@pytest.mark.asyncio
async def test_a_node_removed_since_the_snapshot_is_stale(snapshot, text, source_nodes):
    source_nodes.clear()
    objects = await CodeCompletionRetriever(text, {"operation": "ask"}).get_retrieved_objects(
        "verify_signature"
    )
    assert objects["code_evidence"]["stale_source_ids"] == ["verify"]


@pytest.mark.asyncio
async def test_merged_lanes_union_their_stale_ids(snapshot, text):
    retriever = CodeCompletionRetriever(text, {"operation": "ask"})
    first = await retriever.get_retrieved_objects("verify_signature")
    second = await retriever.get_retrieved_objects("verify_signature")
    first["code_evidence"]["stale_source_ids"] = ["a"]
    second["code_evidence"]["stale_source_ids"] = ["b"]
    merged = retriever.merge_retrieved_objects(first, second)
    assert merged["code_evidence"]["stale_source_ids"] == ["a", "b"]


def test_a_code_query_never_loads_source_nodes():
    from cognee.modules.retrieval.code_retriever import CODE_NODE_TYPES

    assert "CodeSource" not in CODE_NODE_TYPES


def test_a_node_cited_by_both_lanes_appears_once_and_keeps_its_text(snapshot, text):
    retriever = CodeCompletionRetriever(text, {"operation": "ask"})
    outer = _source("class", 1, 20, text="class")
    inner = _source("method", 5, 9, text="method")

    def objects(sources):
        return {
            "code_evidence": {
                "operations": [],
                "sources": sources,
                "question_plan": None,
                "stale_source_ids": [],
            },
            "text_objects": None,
            "text_context": "",
        }

    merged = retriever.merge_retrieved_objects(
        objects(CodeCompletionRetriever._without_nested_text([outer, inner])),
        objects([inner]),
    )
    sources = merged["code_evidence"]["sources"]
    assert [s["node_id"] for s in sources].count("method") == 1
    assert next(s for s in sources if s["node_id"] == "method")["source"]["within"] == "class"


def test_a_node_missing_in_one_lane_and_found_in_the_other_is_cited_with_its_text(snapshot, text):
    retriever = CodeCompletionRetriever(text, {"operation": "ask"})

    def objects(source):
        return {
            "code_evidence": {
                "operations": [],
                "sources": [source],
                "question_plan": None,
                "stale_source_ids": [],
            },
            "text_objects": None,
            "text_context": "",
        }

    merged = retriever.merge_retrieved_objects(
        objects(_source("m", 5, 9, text=None)), objects(_source("m", 5, 9, text="method"))
    )
    (source,) = merged["code_evidence"]["sources"]
    assert source["source"]["text"] == "method"
    assert merged["code_evidence"]["missing_source_ids"] == []

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.session.session_manager import SessionTurnPreparation
from cognee.modules.search.models.EvidenceReference import EvidenceReference
from cognee.modules.search.methods.get_retriever_output import (
    _count_retrieved_objects,
    get_retriever_output,
)
from cognee.modules.search.types import SearchType

# Resolve the module object explicitly. The package __init__ re-exports the
# `get_retriever_output` function under the same name as this submodule, so a
# dotted-string patch target ("...methods.get_retriever_output.<attr>") resolves
# to the function rather than the module and fails. Patching the module object
# directly via patch.object is order-independent and reliable.
get_retriever_output_module = importlib.import_module(
    "cognee.modules.search.methods.get_retriever_output"
)


class _FakeGraphEngine:
    async def is_empty(self):
        return False


class _EffectiveQueryRetriever:
    def __init__(self):
        self.preparation = SessionTurnPreparation(
            should_answer=True,
            effective_query="What should I audit in Lisbon?",
        )
        self.retrieved_query = None
        self.context_query = None
        self.completion_kwargs = None

    async def prepare_session_turn_for_retrieval(self, query):
        assert query == "That was wrong. What should I audit in Lisbon?"
        return self.preparation

    async def get_retrieved_objects(self, query):
        self.retrieved_query = query
        return [{"id": "obj-1"}]

    async def get_context_from_objects(self, query, retrieved_objects):
        self.context_query = query
        assert retrieved_objects == [{"id": "obj-1"}]
        return "context"

    async def get_completion_from_context(
        self,
        query,
        retrieved_objects,
        context,
        effective_query=None,
        turn_preparation=None,
    ):
        self.completion_kwargs = {
            "query": query,
            "effective_query": effective_query,
            "turn_preparation": turn_preparation,
            "context": context,
        }
        return ["answer"]


class _NoAnswerRetriever:
    async def prepare_session_turn_for_retrieval(self, query):
        return SessionTurnPreparation(
            should_answer=False,
            response_to_user="Thanks, I noted that.",
            effective_query=query,
        )

    async def get_retrieved_objects(self, query):
        raise AssertionError("retrieval should be skipped")


class _EvidenceRetriever(_EffectiveQueryRetriever):
    def get_context_evidence(self, retrieved_objects, dataset_id=None):
        assert retrieved_objects == [{"id": "obj-1"}]
        return [
            EvidenceReference(
                kind="segment",
                artifact_id="obj-1",
                dataset_id=str(dataset_id),
                chunk_id="obj-1",
                rank=0,
            )
        ]


class _GraphEvidenceRetriever(_EffectiveQueryRetriever):
    def __init__(self, edge_id):
        super().__init__()
        self.edge_id = edge_id

    def get_context_evidence(self, retrieved_objects, dataset_id=None):
        return [
            EvidenceReference(
                kind="graph_edge",
                artifact_id=str(self.edge_id),
                dataset_id=str(dataset_id),
            )
        ]


@pytest.mark.asyncio
async def test_get_retriever_output_uses_effective_query_before_retrieval():
    retriever = _EffectiveQueryRetriever()
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ),
    ):
        result = await get_retriever_output(
            SearchType.CHUNKS,
            "That was wrong. What should I audit in Lisbon?",
        )

    assert retriever.retrieved_query == "What should I audit in Lisbon?"
    assert retriever.context_query == "What should I audit in Lisbon?"
    assert retriever.completion_kwargs == {
        "query": "That was wrong. What should I audit in Lisbon?",
        "effective_query": "What should I audit in Lisbon?",
        "turn_preparation": retriever.preparation,
        "context": "context",
    }
    assert result.completion == ["answer"]


@pytest.mark.asyncio
async def test_get_retriever_output_skips_retrieval_for_no_answer_turn():
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=_NoAnswerRetriever(),
        ),
    ):
        result = await get_retriever_output(SearchType.CHUNKS, "That was wrong.")

    assert result.result_object is None
    assert result.context is None
    assert result.completion == ["Thanks, I noted that."]


@pytest.mark.asyncio
async def test_get_retriever_output_attaches_structured_context_evidence():
    retriever = _EvidenceRetriever()
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, name="reports", tenant_id=uuid4())
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ),
    ):
        result = await get_retriever_output(
            SearchType.RAG_COMPLETION,
            "That was wrong. What should I audit in Lisbon?",
            dataset=dataset,
            include_references=True,
        )

    assert len(result.evidence) == 1
    assert result.evidence[0].artifact_id == "obj-1"
    assert result.evidence[0].dataset_id == str(dataset_id)


@pytest.mark.asyncio
async def test_get_retriever_output_skips_evidence_hook_when_references_disabled():
    retriever = _EvidenceRetriever()
    retriever.get_context_evidence = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("evidence hook should not be called")
    )
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ),
    ):
        result = await get_retriever_output(
            SearchType.RAG_COMPLETION,
            "That was wrong. What should I audit in Lisbon?",
            include_references=False,
        )

    assert result.evidence == []


@pytest.mark.asyncio
async def test_get_retriever_output_appends_graph_source_evidence():
    edge_id = uuid4()
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, name="reports", tenant_id=uuid4())
    source_reference = EvidenceReference(
        kind="segment",
        artifact_id=str(uuid4()),
        role="supports_assertion",
        assertion_id=str(edge_id),
        dataset_id=str(dataset_id),
    )
    retriever = _GraphEvidenceRetriever(edge_id)
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ),
        patch.object(
            get_retriever_output_module,
            "graph_source_evidence",
            new_callable=AsyncMock,
            return_value=[source_reference],
        ) as resolve_sources,
    ):
        result = await get_retriever_output(
            SearchType.GRAPH_COMPLETION,
            "That was wrong. What should I audit in Lisbon?",
            dataset=dataset,
            include_references=True,
        )

    resolve_sources.assert_awaited_once()
    assert [reference.role for reference in result.evidence] == [
        "used_as_context",
        "supports_assertion",
    ]
    assert result.completion[0].endswith("Evidence:\n- chunk unknown of document unknown")


def test_count_retrieved_objects_counts_structured_lists():
    assert _count_retrieved_objects({"chunks": [1, 2], "entities": [3]}) == 3


def test_count_retrieved_objects_preserves_existing_shapes():
    assert _count_retrieved_objects(None) == 0
    assert _count_retrieved_objects(["a", "b"]) == 2
    assert _count_retrieved_objects({"triplets": []}) == 0
    assert _count_retrieved_objects({"metadata": "value"}) == 1
    assert _count_retrieved_objects("answer") == 1

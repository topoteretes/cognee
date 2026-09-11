import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.session.session_manager import SessionTurnPreparation
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.context_preview import ContextPreview
from cognee.modules.retrieval.session_aware_completion import count_retrieved_objects
from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.models.EvidenceReference import EvidenceReference
from cognee.modules.search.types import ContextFormat, SearchType

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


class _DeterministicRetriever:
    supports_session_turn_preparation = False

    async def prepare_session_turn_for_retrieval(self, query):
        raise AssertionError("deterministic retrieval must not prepare a conversational turn")

    async def get_retrieved_objects(self, query):
        return {"operation": "query_facts", "facts": []}

    async def get_context_from_objects(self, query, retrieved_objects):
        return '{"facts":[],"operation":"query_facts"}'

    async def get_completion_from_context(self, query, retrieved_objects, context):
        return retrieved_objects


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
async def test_get_retriever_output_can_bypass_session_preparation_without_only_context():
    retriever = _DeterministicRetriever()
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
        result = await get_retriever_output(SearchType.CODE, "Checkout")

    assert result.completion == {"operation": "query_facts", "facts": []}


@pytest.mark.asyncio
async def test_get_retriever_output_maps_door_result_without_extra_logic():
    retriever = _NoAnswerRetriever()
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
            "run_session_aware_completion",
            new_callable=AsyncMock,
            return_value=([{"id": "obj-1"}], "context", ["answer"]),
        ) as run_door,
    ):
        result = await get_retriever_output(SearchType.RAG_COMPLETION, "question")

    assert result.result_object == [{"id": "obj-1"}]
    assert result.context == "context"
    assert result.completion == ["answer"]
    run_door.assert_awaited_once_with(
        retriever,
        raw_query="question",
        original_search_type=SearchType.RAG_COMPLETION,
        only_context=False,
        search_type_for_spans=SearchType.RAG_COMPLETION,
    )


class _OnlyContextRetriever:
    """A prompt-carrying retriever whose completion must never be reached."""

    supports_session_turn_preparation = True
    user_prompt_path = "graph_context_for_question.txt"
    system_prompt_path = "answer_simple_question.txt"
    system_prompt = None
    session_id = "session-1"

    async def prepare_session_turn_for_retrieval(self, query):
        raise AssertionError("only_context must not run the pre-retrieval turn analysis")

    async def get_retrieved_objects(self, query):
        return [{"id": "obj-1"}]

    async def get_context_from_objects(self, query, retrieved_objects):
        return "node1 -- rel -- node2"

    async def get_completion_from_context(self, **kwargs):
        raise AssertionError("only_context must not generate a completion")


def _only_context_patches(retriever):
    return (
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
    )


@pytest.mark.asyncio
async def test_only_context_default_format_builds_no_preview():
    """Existing only_context callers must pay nothing and see nothing new."""
    retriever = _OnlyContextRetriever()
    graph_patch, retriever_patch = _only_context_patches(retriever)
    with (
        graph_patch,
        retriever_patch,
        patch.object(
            get_retriever_output_module,
            "build_context_preview",
            new_callable=AsyncMock,
        ) as preview,
    ):
        result = await get_retriever_output(SearchType.GRAPH_COMPLETION, "why?", only_context=True)

    preview.assert_not_awaited()
    assert result.context == "node1 -- rel -- node2"
    assert result.result == "node1 -- rel -- node2"
    assert result.session_context is None
    assert result.user_prompt is None


@pytest.mark.asyncio
async def test_only_context_prompt_format_populates_the_envelope():
    retriever = _OnlyContextRetriever()
    graph_patch, retriever_patch = _only_context_patches(retriever)
    with (
        graph_patch,
        retriever_patch,
        patch.object(
            get_retriever_output_module,
            "build_context_preview",
            new_callable=AsyncMock,
            return_value=ContextPreview(
                session_context="## Active session guidance\n- be terse",
                user_prompt="The question is: `why?`",
                system_prompt="history\nTASK:answer",
            ),
        ) as preview,
    ):
        result = await get_retriever_output(
            SearchType.GRAPH_COMPLETION,
            "why?",
            only_context=True,
            context_format=ContextFormat.PROMPT,
        )

    assert preview.await_args.kwargs == {
        "query": "why?",
        "context": "node1 -- rel -- node2",
        "session_id": None,
        "shared_history": None,
    }
    assert result.question == "why?"
    assert result.session_context == "## Active session guidance\n- be terse"
    assert result.user_prompt == "The question is: `why?`"
    assert result.system_prompt == "history\nTASK:answer"
    assert result.result["question"] == "why?"
    assert result.result["context"] == "node1 -- rel -- node2"


@pytest.mark.asyncio
async def test_prompt_format_is_ignored_when_only_context_is_off():
    """A normal completion already sends the prompt; there is nothing to preview."""
    retriever = _DeterministicRetriever()
    graph_patch, retriever_patch = _only_context_patches(retriever)
    with (
        graph_patch,
        retriever_patch,
        patch.object(
            get_retriever_output_module,
            "build_context_preview",
            new_callable=AsyncMock,
        ) as preview,
    ):
        result = await get_retriever_output(
            SearchType.CODE, "Checkout", context_format=ContextFormat.PROMPT
        )

    preview.assert_not_awaited()
    assert result.completion == {"operation": "query_facts", "facts": []}


@pytest.mark.asyncio
async def test_unknown_context_format_is_rejected_before_retrieval():
    """One shared rule at every entry point: the same error, and no wasted retrieval."""
    retriever = _OnlyContextRetriever()
    graph_patch, retriever_patch = _only_context_patches(retriever)
    with graph_patch, retriever_patch as factory:
        with pytest.raises(CogneeValidationError) as excinfo:
            await get_retriever_output(
                SearchType.GRAPH_COMPLETION, "why?", only_context=True, context_format="bogus"
            )

    assert excinfo.value.name == "InvalidContextFormatError"
    factory.assert_not_awaited()


@pytest.mark.asyncio
async def test_prompt_preview_receives_the_fan_outs_shared_history():
    retriever = _OnlyContextRetriever()
    graph_patch, retriever_patch = _only_context_patches(retriever)
    shared = object()
    with (
        graph_patch,
        retriever_patch,
        patch.object(
            get_retriever_output_module,
            "build_context_preview",
            new_callable=AsyncMock,
            return_value=ContextPreview(),
        ) as preview,
    ):
        await get_retriever_output(
            SearchType.GRAPH_COMPLETION,
            "why?",
            only_context=True,
            context_format="prompt",
            session_id="s1",
            shared_history=shared,
        )

    assert preview.await_args.kwargs["session_id"] == "s1"
    assert preview.await_args.kwargs["shared_history"] is shared


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
    assert count_retrieved_objects({"chunks": [1, 2], "entities": [3]}) == 3


def test_count_retrieved_objects_preserves_existing_shapes():
    assert count_retrieved_objects(None) == 0
    assert count_retrieved_objects(["a", "b"]) == 2
    assert count_retrieved_objects({"triplets": []}) == 0
    assert count_retrieved_objects({"metadata": "value"}) == 1
    assert count_retrieved_objects("answer") == 1


@pytest.mark.asyncio
async def test_hybrid_deferral_reports_graph_completion():
    retriever = _DeterministicRetriever()
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "hybrid_deferral_reason",
            new_callable=AsyncMock,
            return_value="Entity_name collection missing",
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ) as factory,
        patch.object(
            get_retriever_output_module,
            "run_session_aware_completion",
            new_callable=AsyncMock,
            return_value=({"chunks": []}, "context", ["answer"]),
        ),
    ):
        result = await get_retriever_output(SearchType.HYBRID_COMPLETION, "q")

    assert result.search_type is SearchType.GRAPH_COMPLETION
    assert factory.await_args.kwargs["query_type"] is SearchType.GRAPH_COMPLETION


@pytest.mark.asyncio
async def test_feeling_lucky_hybrid_deferral_reports_graph_completion():
    retriever = _DeterministicRetriever()
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "select_search_type",
            new_callable=AsyncMock,
            return_value=SearchType.HYBRID_COMPLETION,
        ),
        patch.object(
            get_retriever_output_module,
            "hybrid_deferral_reason",
            new_callable=AsyncMock,
            return_value="neighborhood_depth is set",
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ) as factory,
        patch.object(
            get_retriever_output_module,
            "run_session_aware_completion",
            new_callable=AsyncMock,
            return_value=({"chunks": []}, "context", ["answer"]),
        ),
    ):
        result = await get_retriever_output(SearchType.FEELING_LUCKY, "q", neighborhood_depth=2)

    assert result.search_type is SearchType.GRAPH_COMPLETION
    assert factory.await_args.kwargs["query_type"] is SearchType.GRAPH_COMPLETION


@pytest.mark.asyncio
async def test_feeling_lucky_on_empty_graph_skips_selector_and_keeps_hybrid():
    class _EmptyGraph:
        async def is_empty(self):
            return True

    retriever = _DeterministicRetriever()
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_EmptyGraph(),
        ),
        patch.object(
            get_retriever_output_module,
            "select_search_type",
            new_callable=AsyncMock,
        ) as selector,
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ) as factory,
        patch.object(
            get_retriever_output_module,
            "run_session_aware_completion",
            new_callable=AsyncMock,
            return_value=(
                {"chunks": [], "chunk_summaries": {}, "entities": [], "facts": []},
                "",
                ["answer"],
            ),
        ),
    ):
        result = await get_retriever_output(SearchType.FEELING_LUCKY, "q")

    selector.assert_not_awaited()
    assert result.search_type is SearchType.HYBRID_COMPLETION
    assert factory.await_args.kwargs["query_type"] is SearchType.HYBRID_COMPLETION


class _Entity:
    pass


def _factory_and_session_patches(retriever):
    return (
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
            "run_session_aware_completion",
            new_callable=AsyncMock,
            return_value=({"chunks": []}, "context", ["answer"]),
        ),
    )


@pytest.mark.asyncio
async def test_non_nodeset_scope_defers_and_forwards_node_name():
    retriever = _DeterministicRetriever()
    graph, factory, session = _factory_and_session_patches(retriever)
    with graph, factory as factory_mock, session:
        result = await get_retriever_output(
            SearchType.HYBRID_COMPLETION,
            "q",
            node_name=["KEN"],
            node_type=_Entity,
        )

    assert result.search_type is SearchType.GRAPH_COMPLETION
    assert factory_mock.await_args.kwargs["query_type"] is SearchType.GRAPH_COMPLETION
    assert factory_mock.await_args.kwargs["node_name"] == ["KEN"]
    assert factory_mock.await_args.kwargs["node_type"] is _Entity


@pytest.mark.asyncio
async def test_nodeset_scope_stays_on_hybrid_and_forwards_node_name():
    retriever = _DeterministicRetriever()
    vector = AsyncMock()
    vector.has_collection = AsyncMock(return_value=True)
    graph, factory, session = _factory_and_session_patches(retriever)
    with (
        graph,
        factory as factory_mock,
        session,
        patch(
            "cognee.modules.search.methods.hybrid_deferral.get_vector_engine_async",
            new_callable=AsyncMock,
            return_value=vector,
        ),
    ):
        result = await get_retriever_output(
            SearchType.HYBRID_COMPLETION,
            "q",
            node_name=["KEN"],
            node_type=NodeSet,
        )

    assert result.search_type is SearchType.HYBRID_COMPLETION
    assert factory_mock.await_args.kwargs["query_type"] is SearchType.HYBRID_COMPLETION
    assert factory_mock.await_args.kwargs["node_name"] == ["KEN"]
    assert factory_mock.await_args.kwargs["node_type"] is NodeSet


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wide_search_top_k": 200}, "wide_search_top_k"),
        ({"triplet_distance_penalty": 2.5}, "triplet_distance_penalty"),
    ],
)
@pytest.mark.asyncio
async def test_hybrid_rejects_graph_only_knobs(kwargs, match):
    retriever = _DeterministicRetriever()
    graph, factory, session = _factory_and_session_patches(retriever)
    with graph, factory as factory_mock, session:
        with pytest.raises(CogneeValidationError, match=match):
            await get_retriever_output(SearchType.HYBRID_COMPLETION, "q", **kwargs)

    factory_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_completion_accepts_graph_only_knobs():
    retriever = _DeterministicRetriever()
    graph, factory, session = _factory_and_session_patches(retriever)
    with graph, factory as factory_mock, session:
        result = await get_retriever_output(
            SearchType.GRAPH_COMPLETION,
            "q",
            wide_search_top_k=200,
            triplet_distance_penalty=2.5,
        )

    assert result.search_type is SearchType.GRAPH_COMPLETION
    assert factory_mock.await_args.kwargs["wide_search_top_k"] == 200
    assert factory_mock.await_args.kwargs["triplet_distance_penalty"] == 2.5

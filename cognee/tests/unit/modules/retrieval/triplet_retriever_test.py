from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.session.session_manager import SessionTurnPreparation
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.tasks.storage.add_data_points import _create_triplets_from_graph


@pytest.fixture(autouse=True)
def _no_real_llm_calls():
    """Keep every test in this file off the network.

    ``get_completion_from_context`` calls ``generate_completion``, which
    test_get_context_success does not patch, so the call escaped to the real
    provider once the suite was sharded and the accidental upstream patch
    was no longer in the same process. Per-test patches override this one.
    """
    with patch(
        "cognee.modules.retrieval.triplet_retriever.generate_completion",
        new=AsyncMock(return_value="Generated answer"),
    ):
        yield


@pytest.fixture
def mock_vector_engine():
    """Create a mock vector engine."""
    engine = AsyncMock()
    engine.has_collection = AsyncMock(return_value=True)
    engine.search = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_get_context_success(mock_vector_engine):
    """Test successful retrieval of triplet context."""
    mock_result1 = MagicMock()
    mock_result1.payload = {"text": "Alice knows Bob"}
    mock_result2 = MagicMock()
    mock_result2.payload = {"text": "Bob works at Tech Corp"}

    mock_vector_engine.search.return_value = [mock_result1, mock_result2]

    retriever = TripletRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", objects)
        await retriever.get_completion_from_context("test query", objects, context)

    assert context == "Alice knows Bob\nBob works at Tech Corp"
    mock_vector_engine.search.assert_awaited_once_with(
        "Triplet_text",
        "test query",
        limit=5,
        include_payload=True,
    )


@pytest.mark.asyncio
async def test_get_objects_no_collection(mock_vector_engine):
    """Test that NoDataError is raised when Triplet_text collection doesn't exist."""
    mock_vector_engine.has_collection.return_value = False

    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        pytest.raises(NoDataError, match="create_triplet_embeddings"),
    ):
        await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_get_context_empty_results(mock_vector_engine):
    """Test that empty string is returned when no triplets are found."""
    mock_vector_engine.search.return_value = []

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context_from_objects("test query", [])

    assert context == ""


@pytest.mark.asyncio
async def test_get_objects_collection_not_found_error(mock_vector_engine):
    """Test that CollectionNotFoundError is converted to NoDataError."""
    mock_vector_engine.has_collection.side_effect = CollectionNotFoundError("Collection not found")

    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        pytest.raises(NoDataError, match="No data found"),
    ):
        await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_get_context_empty_payload_text(mock_vector_engine):
    """Test get_context handles missing text in payload."""
    mock_result = MagicMock()
    mock_result.payload = {}

    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        pytest.raises(KeyError),
    ):
        objects = await retriever.get_retrieved_objects("test query")
        await retriever.get_context_from_objects("test query", retrieved_objects=objects)


@pytest.mark.asyncio
async def test_get_context_single_triplet(mock_vector_engine):
    """Test get_context with single triplet result."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Single triplet"}

    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects=objects)

    assert context == "Single triplet"


@pytest.mark.asyncio
async def test_init_defaults():
    """Test TripletRetriever initialization with defaults."""
    retriever = TripletRetriever()

    assert retriever.user_prompt_path == "context_for_question.txt"
    assert retriever.system_prompt_path == "answer_simple_question.txt"
    assert retriever.top_k == 5  # Default is 5
    assert retriever.system_prompt is None
    assert retriever.node_name is None
    assert retriever.node_name_filter_operator == "OR"


@pytest.mark.asyncio
async def test_init_custom_params():
    """Test TripletRetriever initialization with custom parameters."""
    retriever = TripletRetriever(
        user_prompt_path="custom_user.txt",
        system_prompt_path="custom_system.txt",
        system_prompt="Custom prompt",
        top_k=10,
        node_name=["KEN", "src_type:figure"],
        node_name_filter_operator="AND",
    )

    assert retriever.user_prompt_path == "custom_user.txt"
    assert retriever.system_prompt_path == "custom_system.txt"
    assert retriever.system_prompt == "Custom prompt"
    assert retriever.top_k == 10
    assert retriever.node_name == ["KEN", "src_type:figure"]
    assert retriever.node_name_filter_operator == "AND"


def _edge(source_id, target_id, relationship_name):
    return (source_id, target_id, relationship_name, {})


def _triplet_id(source_id, relationship_name, target_id):
    return str(generate_node_id(source_id + relationship_name + target_id))


@pytest.mark.asyncio
async def test_node_set_scope_ranks_only_triplets_of_in_scope_edges(mock_vector_engine):
    """A node-set search scores the triplets of the subgraph's edges instead of filtering
    Triplet_text by a payload tag triplets never carry (gh #4822)."""
    near, far = _triplet_id("a", "works_at", "b"), _triplet_id("b", "located_in", "c")
    graph_engine = AsyncMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(
        return_value=([], [_edge("a", "b", "works_at"), _edge("b", "c", "located_in")])
    )
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2]])
    mock_vector_engine.score_by_ids = AsyncMock(
        return_value=[
            ScoredResult(id=far, score=0.9, payload=None),
            ScoredResult(id=near, score=0.1, payload=None),
        ]
    )
    mock_vector_engine.retrieve = AsyncMock(
        return_value=[ScoredResult(id=near, score=0, payload={"text": "a works at b"})]
    )

    retriever = TripletRetriever(top_k=1, node_name=["A"], node_name_filter_operator="AND")

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_graph_engine",
            AsyncMock(return_value=graph_engine),
        ),
    ):
        objects = await retriever.get_retrieved_objects("where does a work?")

    graph_engine.get_nodeset_subgraph.assert_awaited_once_with(
        node_type=NodeSet, node_name=["A"], node_name_filter_operator="AND"
    )
    assert sorted(mock_vector_engine.score_by_ids.await_args.args[1]) == sorted([near, far])
    mock_vector_engine.retrieve.assert_awaited_once_with("Triplet_text", [near])
    mock_vector_engine.search.assert_not_awaited()
    assert [(str(o.id), o.score, o.payload["text"]) for o in objects] == [
        (near, 0.1, "a works at b")
    ]


@pytest.mark.asyncio
async def test_node_set_scope_with_empty_subgraph_returns_nothing(mock_vector_engine):
    graph_engine = AsyncMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(return_value=([], []))
    mock_vector_engine.score_by_ids = AsyncMock()

    retriever = TripletRetriever(node_name=["missing"])

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_graph_engine",
            AsyncMock(return_value=graph_engine),
        ),
    ):
        assert await retriever.get_retrieved_objects("anything") == []

    mock_vector_engine.score_by_ids.assert_not_awaited()
    mock_vector_engine.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_completion_without_context(mock_vector_engine):
    """No context means no LLM call and no results (SDK-270 / gh #3728):
    get_completion_from_context never re-derives a missing context."""
    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            new_callable=AsyncMock,
        ) as mock_generate,
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context("test query", None, None)

    assert completion == []
    mock_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_completion_with_provided_context(mock_vector_engine):
    """Test get_completion uses provided context."""
    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context(
            "test query", None, context="Provided context"
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_session(mock_vector_engine):
    """Test get_completion with session caching enabled (SessionManager path)."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Test triplet"}
    mock_vector_engine.has_collection.return_value = True
    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever(session_id="test_session")

    mock_user = MagicMock()
    mock_user.id = "test-user-id"

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_session_manager",
        ) as mock_get_sm,
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
        patch("cognee.modules.retrieval.triplet_retriever.session_user") as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user
        mock_sm = MagicMock()
        mock_sm.generate_completion_with_session = AsyncMock(return_value="Generated answer")
        mock_get_sm.return_value = mock_sm

        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects=objects)
        turn_preparation = SessionTurnPreparation(effective_query="prepared query")
        completion = await retriever.get_completion_from_context(
            "test query",
            objects,
            context,
            effective_query="prepared query",
            turn_preparation=turn_preparation,
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
    mock_sm.generate_completion_with_session.assert_awaited_once()
    call_kw = mock_sm.generate_completion_with_session.call_args.kwargs
    assert call_kw.get("used_graph_element_ids") is None
    assert call_kw["effective_query"] == "prepared query"
    assert call_kw["turn_preparation"] is turn_preparation


@pytest.mark.asyncio
async def test_get_completion_with_session_no_user_id(mock_vector_engine):
    """Test get_completion with session config but no user ID."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Test triplet"}
    mock_vector_engine.has_collection.return_value = True
    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
        patch("cognee.modules.retrieval.triplet_retriever.session_user") as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = None  # No user

        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects=objects)
        completion = await retriever.get_completion_from_context("test query", objects, context)

    assert isinstance(completion, list)
    assert len(completion) == 1


@pytest.mark.asyncio
async def test_get_completion_with_response_model(mock_vector_engine):
    """Test get_completion with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    mock_result = MagicMock()
    mock_result.payload = {"text": "Test triplet"}
    mock_vector_engine.has_collection.return_value = True
    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever(response_model=TestModel)

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            return_value=TestModel(answer="Test answer"),
        ),
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects=objects)
        completion = await retriever.get_completion_from_context("test query", objects, context)

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert isinstance(completion[0], TestModel)


@pytest.mark.asyncio
async def test_init_none_top_k():
    """Test TripletRetriever initialization with None top_k."""
    retriever = TripletRetriever(top_k=None)

    assert retriever.top_k == 5


@pytest.mark.asyncio
async def test_node_set_scope_derives_the_triplet_id_the_writer_stores(mock_vector_engine):
    """The scoped search finds triplets only by recomputing their ids from the edges, so its
    id must match the one _create_triplets_from_graph writes for the same edge."""
    alice = Entity(name="Alice", description="person")
    acme = Entity(name="Acme", description="company")
    edge = (str(alice.id), str(acme.id), "works_at", {"edge_text": "Alice works at Acme"})
    [written] = _create_triplets_from_graph([alice, acme], [edge])

    graph_engine = AsyncMock()
    graph_engine.get_nodeset_subgraph = AsyncMock(return_value=([], [edge]))
    mock_vector_engine.embedding_engine = AsyncMock()
    mock_vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2]])
    mock_vector_engine.score_by_ids = AsyncMock(return_value=[])

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_graph_engine",
            AsyncMock(return_value=graph_engine),
        ),
    ):
        await TripletRetriever(node_name=["A"]).get_retrieved_objects("where does Alice work?")

    assert mock_vector_engine.score_by_ids.await_args.args[1] == [str(written.id)]

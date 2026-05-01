import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.low_level import DataPoint
from cognee.low_level import setup as cognee_setup
from cognee.modules.retrieval.graph_completion_decomposition_retriever import (
    GraphCompletionDecompositionRetriever,
    QueryDecomposition,
)
from cognee.tasks.storage import add_data_points

ORIGINAL_QUERY = "Who works at Figma and who works at Canva?"
SUBQUERIES = ["Who works at Figma?", "Who works at Canva?"]


def _clear_engine_caches() -> None:
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()


def _combined_decomposition() -> QueryDecomposition:
    return QueryDecomposition(subqueries=SUBQUERIES)


async def _fake_combined_mode_llm(text_input: str, system_prompt: str, response_model, **kwargs):
    if response_model is QueryDecomposition:
        return _combined_decomposition()
    if ORIGINAL_QUERY in text_input:
        return "Combined answer for both companies."
    return "Unused combined answer"


async def _fake_answer_per_subquery_llm(
    text_input: str, system_prompt: str, response_model, **kwargs
):
    if response_model is QueryDecomposition:
        return _combined_decomposition()
    if ORIGINAL_QUERY in text_input:
        return (
            "Figma employees are Steve Rodger, Ike Loma, and Jason Statham. "
            "Canva employees are Mike Broski and Christina Mayer."
        )
    if SUBQUERIES[0] in text_input:
        return "Steve Rodger, Ike Loma, and Jason Statham work at Figma."
    if SUBQUERIES[1] in text_input:
        return "Mike Broski and Christina Mayer work at Canva."
    return "Unexpected answer"


@pytest_asyncio.fixture
async def setup_test_environment_simple():
    """Set up a clean test environment with simple graph data."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(
        base_dir / ".cognee_system/test_graph_completion_decomposition_context_simple"
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_graph_completion_decomposition_context_simple"
    )

    cognee.config.set_graph_database_provider("kuzu")
    cognee.config.set_vector_db_config({"vector_db_provider": "lancedb"})
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    _clear_engine_caches()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    _clear_engine_caches()
    await cognee_setup()

    class Company(DataPoint):
        name: str

    class Person(DataPoint):
        name: str
        works_for: Company

    company1 = Company(name="Figma")
    company2 = Company(name="Canva")
    person1 = Person(name="Steve Rodger", works_for=company1)
    person2 = Person(name="Ike Loma", works_for=company1)
    person3 = Person(name="Jason Statham", works_for=company1)
    person4 = Person(name="Mike Broski", works_for=company2)
    person5 = Person(name="Christina Mayer", works_for=company2)

    await add_data_points([company1, company2, person1, person2, person3, person4, person5])

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        _clear_engine_caches()
    except Exception:
        pass


@pytest_asyncio.fixture
async def setup_test_environment_empty():
    """Set up a clean test environment without graph data."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(
        base_dir / ".cognee_system/test_graph_completion_decomposition_context_empty"
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_graph_completion_decomposition_context_empty"
    )

    cognee.config.set_graph_database_provider("kuzu")
    cognee.config.set_vector_db_config({"vector_db_provider": "lancedb"})
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    _clear_engine_caches()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    _clear_engine_caches()
    await cognee_setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        _clear_engine_caches()
    except Exception:
        pass


@pytest.fixture
def session_manager(tmp_path):
    with patch(
        "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
        return_value={"data_root_directory": str(tmp_path)},
    ):
        from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter

        adapter = FSCacheAdapter()
        yield SessionManager(cache_engine=adapter)
        adapter.cache.close()


@pytest.mark.asyncio
async def test_graph_completion_decomposition_combined_mode_context_coverage(
    setup_test_environment_simple,
):
    retriever = GraphCompletionDecompositionRetriever(
        decomposition_mode="combined_triplets_context",
        top_k=20,
    )

    with patch(
        "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=_fake_combined_mode_llm,
    ):
        triplets = await retriever.get_retrieved_objects(ORIGINAL_QUERY)
        context = await retriever.get_context_from_objects(
            query=ORIGINAL_QUERY,
            retrieved_objects=triplets,
        )
        answer = await retriever.get_completion_from_context(
            query=ORIGINAL_QUERY,
            retrieved_objects=triplets,
            context=context,
        )

    assert "Steve Rodger --[works_for]--> Figma" in context
    assert "Ike Loma --[works_for]--> Figma" in context
    assert "Mike Broski --[works_for]--> Canva" in context
    assert "Christina Mayer --[works_for]--> Canva" in context
    assert answer == ["Combined answer for both companies."]


@pytest.mark.asyncio
async def test_graph_completion_decomposition_answer_per_subquery_synthesis(
    setup_test_environment_simple,
):
    retriever = GraphCompletionDecompositionRetriever(top_k=20)

    with patch(
        "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output",
        new_callable=AsyncMock,
        side_effect=_fake_answer_per_subquery_llm,
    ):
        triplets = await retriever.get_retrieved_objects(ORIGINAL_QUERY)
        context = await retriever.get_context_from_objects(
            query=ORIGINAL_QUERY,
            retrieved_objects=triplets,
        )
        answer = await retriever.get_completion_from_context(
            query=ORIGINAL_QUERY,
            retrieved_objects=triplets,
            context=context,
        )

    assert "Question decomposition results:" in context
    assert "Subquery 1: Who works at Figma?" in context
    assert "Subquery 2: Who works at Canva?" in context
    assert "Steve Rodger, Ike Loma, and Jason Statham work at Figma." in context
    assert "Mike Broski and Christina Mayer work at Canva." in context
    assert answer == [
        (
            "Figma employees are Steve Rodger, Ike Loma, and Jason Statham. "
            "Canva employees are Mike Broski and Christina Mayer."
        )
    ]


@pytest.mark.asyncio
async def test_graph_completion_decomposition_context_empty_graph(setup_test_environment_empty):
    retriever = GraphCompletionDecompositionRetriever()

    with patch(
        "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output",
        new_callable=AsyncMock,
        return_value=_combined_decomposition(),
    ):
        triplets = await retriever.get_retrieved_objects(ORIGINAL_QUERY)
        context = await retriever.get_context_from_objects(
            query=ORIGINAL_QUERY,
            retrieved_objects=triplets,
        )

    assert triplets == []
    assert context == ""


@pytest.mark.asyncio
async def test_graph_completion_decomposition_combined_mode_session_stores_only_final_qa(
    setup_test_environment_simple,
    session_manager,
):
    retriever = GraphCompletionDecompositionRetriever(
        decomposition_mode="combined_triplets_context",
        top_k=20,
        session_id="session-1",
    )
    user = SimpleNamespace(id="user-1")

    with (
        patch(
            "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output",
            new_callable=AsyncMock,
            side_effect=_fake_combined_mode_llm,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_session_manager",
            return_value=session_manager,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_retriever_cache_config,
        patch(
            "cognee.infrastructure.session.session_manager.CacheConfig"
        ) as mock_session_cache_config,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.session_user"
        ) as mock_retriever_session_user,
        patch(
            "cognee.infrastructure.session.session_manager.session_user"
        ) as mock_session_manager_user,
    ):
        retriever_cache_config = MagicMock()
        retriever_cache_config.caching = True
        mock_retriever_cache_config.return_value = retriever_cache_config

        session_cache_config = MagicMock()
        session_cache_config.caching = True
        session_cache_config.auto_feedback = False
        mock_session_cache_config.return_value = session_cache_config

        mock_retriever_session_user.get.return_value = user
        mock_session_manager_user.get.return_value = user

        triplets = await retriever.get_retrieved_objects(ORIGINAL_QUERY)
        context = await retriever.get_context_from_objects(
            query=ORIGINAL_QUERY,
            retrieved_objects=triplets,
        )
        answer = await retriever.get_completion_from_context(
            query=ORIGINAL_QUERY,
            retrieved_objects=triplets,
            context=context,
        )

    assert answer == ["Combined answer for both companies."]

    entries = await session_manager.get_session(user_id="user-1", session_id="session-1")
    assert len(entries) == 1
    assert entries[0].question == ORIGINAL_QUERY
    assert entries[0].answer == "Combined answer for both companies."
    assert entries[0].used_graph_element_ids is not None

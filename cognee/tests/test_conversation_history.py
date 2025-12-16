"""
End-to-end integration test for conversation history feature.

Tests all retrievers that save conversation history to Redis cache:
1. GRAPH_COMPLETION
2. RAG_COMPLETION
3. GRAPH_COMPLETION_COT
4. GRAPH_COMPLETION_CONTEXT_EXTENSION
5. GRAPH_SUMMARY_COMPLETION
6. TEMPORAL
7. TRIPLET_COMPLETION
"""

import os
import cognee
import pathlib

from cognee.infrastructure.databases.cache import get_cache_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from collections import Counter

logger = get_logger()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".data_storage/test_conversation_history",
            )
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".cognee_system/test_conversation_history",
            )
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "conversation_history_test"

    text_1 = """TechCorp is a technology company based in San Francisco. They specialize in artificial intelligence and machine learning."""
    text_2 = (
        """DataCo is a data analytics company. They help businesses make sense of their data."""
    )

    await cognee.add(data=text_1, dataset_name=dataset_name)
    await cognee.add(data=text_2, dataset_name=dataset_name)

    await cognee.cognify(datasets=[dataset_name])

    user = await get_default_user()

    from cognee.memify_pipelines.create_triplet_embeddings import create_triplet_embeddings

    await create_triplet_embeddings(user=user, dataset=dataset_name)

    cache_engine = get_cache_engine()
    assert cache_engine is not None, "Cache engine should be available for testing"

    session_id_1 = "test_session_graph"

    await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is TechCorp?",
        session_id=session_id_1,
    )

    history1 = await cache_engine.get_latest_qa(str(user.id), session_id_1, last_n=10)
    assert len(history1) == 1, f"Expected at least 1 Q&A in history, got {len(history1)}"
    our_qa = [h for h in history1 if h["question"] == "What is TechCorp?"]
    assert len(our_qa) >= 1, "Expected to find 'What is TechCorp?' in history"
    assert "answer" in our_qa[0] and "context" in our_qa[0], (
        "Q&A should contain answer and context fields"
    )

    result2 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Tell me more about it",
        session_id=session_id_1,
    )

    assert isinstance(result2, list) and len(result2) > 0, (
        f"Second query should return non-empty list, got: {result2!r}"
    )

    history2 = await cache_engine.get_latest_qa(str(user.id), session_id_1, last_n=10)
    our_questions = [
        h for h in history2 if h["question"] in ["What is TechCorp?", "Tell me more about it"]
    ]
    assert len(our_questions) == 2, (
        f"Expected at least 2 Q&A pairs in history after 2 queries, got {len(our_questions)}"
    )

    session_id_2 = "test_session_separate"

    result3 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is DataCo?",
        session_id=session_id_2,
    )

    assert isinstance(result3, list) and len(result3) > 0, (
        f"Different session should return non-empty list, got: {result3!r}"
    )

    history3 = await cache_engine.get_latest_qa(str(user.id), session_id_2, last_n=10)
    our_qa_session2 = [h for h in history3 if h["question"] == "What is DataCo?"]
    assert len(our_qa_session2) == 1, "Session 2 should have 'What is DataCo?' question"

    result4 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Test default session",
        session_id=None,
    )

    assert isinstance(result4, list) and len(result4) > 0, (
        f"Default session should return non-empty list, got: {result4!r}"
    )

    history_default = await cache_engine.get_latest_qa(str(user.id), "default_session", last_n=10)
    our_qa_default = [h for h in history_default if h["question"] == "Test default session"]
    assert len(our_qa_default) == 1, "Should find 'Test default session' in default_session"

    session_id_rag = "test_session_rag"

    result_rag = await cognee.search(
        query_type=SearchType.RAG_COMPLETION,
        query_text="What companies are mentioned?",
        session_id=session_id_rag,
    )

    assert isinstance(result_rag, list) and len(result_rag) > 0, (
        f"RAG_COMPLETION should return non-empty list, got: {result_rag!r}"
    )

    history_rag = await cache_engine.get_latest_qa(str(user.id), session_id_rag, last_n=10)
    our_qa_rag = [h for h in history_rag if h["question"] == "What companies are mentioned?"]
    assert len(our_qa_rag) == 1, "Should find RAG question in history"

    session_id_cot = "test_session_cot"

    result_cot = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION_COT,
        query_text="What do you know about TechCorp?",
        session_id=session_id_cot,
    )

    assert isinstance(result_cot, list) and len(result_cot) > 0, (
        f"GRAPH_COMPLETION_COT should return non-empty list, got: {result_cot!r}"
    )

    history_cot = await cache_engine.get_latest_qa(str(user.id), session_id_cot, last_n=10)
    our_qa_cot = [h for h in history_cot if h["question"] == "What do you know about TechCorp?"]
    assert len(our_qa_cot) == 1, "Should find CoT question in history"

    session_id_ext = "test_session_ext"

    result_ext = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        query_text="Tell me about DataCo",
        session_id=session_id_ext,
    )

    assert isinstance(result_ext, list) and len(result_ext) > 0, (
        f"GRAPH_COMPLETION_CONTEXT_EXTENSION should return non-empty list, got: {result_ext!r}"
    )

    history_ext = await cache_engine.get_latest_qa(str(user.id), session_id_ext, last_n=10)
    our_qa_ext = [h for h in history_ext if h["question"] == "Tell me about DataCo"]
    assert len(our_qa_ext) == 1, "Should find Context Extension question in history"

    session_id_summary = "test_session_summary"

    result_summary = await cognee.search(
        query_type=SearchType.GRAPH_SUMMARY_COMPLETION,
        query_text="What are the key points about TechCorp?",
        session_id=session_id_summary,
    )

    assert isinstance(result_summary, list) and len(result_summary) > 0, (
        f"GRAPH_SUMMARY_COMPLETION should return non-empty list, got: {result_summary!r}"
    )

    history_summary = await cache_engine.get_latest_qa(str(user.id), session_id_summary, last_n=10)
    our_qa_summary = [
        h for h in history_summary if h["question"] == "What are the key points about TechCorp?"
    ]
    assert len(our_qa_summary) == 1, "Should find Summary question in history"

    session_id_temporal = "test_session_temporal"

    result_temporal = await cognee.search(
        query_type=SearchType.TEMPORAL,
        query_text="Tell me about the companies",
        session_id=session_id_temporal,
    )

    assert isinstance(result_temporal, list) and len(result_temporal) > 0, (
        f"TEMPORAL should return non-empty list, got: {result_temporal!r}"
    )

    history_temporal = await cache_engine.get_latest_qa(
        str(user.id), session_id_temporal, last_n=10
    )
    our_qa_temporal = [
        h for h in history_temporal if h["question"] == "Tell me about the companies"
    ]
    assert len(our_qa_temporal) == 1, "Should find Temporal question in history"

    session_id_triplet = "test_session_triplet"

    result_triplet = await cognee.search(
        query_type=SearchType.TRIPLET_COMPLETION,
        query_text="What companies are mentioned?",
        session_id=session_id_triplet,
    )

    assert isinstance(result_triplet, list) and len(result_triplet) > 0, (
        f"TRIPLET_COMPLETION should return non-empty list, got: {result_triplet!r}"
    )

    history_triplet = await cache_engine.get_latest_qa(str(user.id), session_id_triplet, last_n=10)
    our_qa_triplet = [
        h for h in history_triplet if h["question"] == "What companies are mentioned?"
    ]
    assert len(our_qa_triplet) == 1, "Should find Triplet question in history"

    from cognee.modules.retrieval.utils.session_cache import (
        get_conversation_history,
    )

    formatted_history = await get_conversation_history(session_id=session_id_1)

    assert "Previous conversation:" in formatted_history, (
        "Formatted history should contain 'Previous conversation:' header"
    )
    assert "QUESTION:" in formatted_history, "Formatted history should contain 'QUESTION:' prefix"
    assert "CONTEXT:" in formatted_history, "Formatted history should contain 'CONTEXT:' prefix"
    assert "ANSWER:" in formatted_history, "Formatted history should contain 'ANSWER:' prefix"

    from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
        persist_sessions_in_knowledge_graph_pipeline,
    )

    logger.info("Starting persist_sessions_in_knowledge_graph tests")

    await persist_sessions_in_knowledge_graph_pipeline(
        user=user,
        session_ids=[session_id_1, session_id_2],
        dataset=dataset_name,
        run_in_background=False,
    )

    graph_engine = await get_graph_engine()
    graph = await graph_engine.get_graph_data()

    type_counts = Counter(node_data[1].get("type", {}) for node_data in graph[0])

    "Tests the correct number of NodeSet nodes after session persistence"
    assert type_counts.get("NodeSet", 0) == 1, (
        f"Number of NodeSets in the graph is incorrect, found {type_counts.get('NodeSet', 0)} but there should be exactly 1."
    )

    "Tests the correct number of DocumentChunk nodes after session persistence"
    assert type_counts.get("DocumentChunk", 0) == 4, (
        f"Number of DocumentChunk ndoes in the graph is incorrect, found {type_counts.get('DocumentChunk', 0)} but there should be exactly 4 (2 original documents, 2 sessions)."
    )

    from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine

    vector_engine = get_vector_engine()
    collection_size = await vector_engine.search(
        collection_name="DocumentChunk_text",
        query_text="test",
        limit=1000,
    )
    assert len(collection_size) == 4, (
        f"DocumentChunk_text collection should have exactly 4 embeddings, found {len(collection_size)}"
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    logger.info("All conversation history tests passed successfully")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

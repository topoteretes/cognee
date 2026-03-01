import asyncio
import os
import cognee
from cognee import visualize_graph
from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
    persist_sessions_in_knowledge_graph_pipeline,
)
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger("conversation_session_persistence_example")


async def main():
    # NOTE: CACHING has to be enabled for this example to work
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text_1 = "Cognee is a solution that can build knowledge graph from text, creating an AI memory system"
    text_2 = "Germany is a country located next to the Netherlands"

    await cognee.add([text_1, text_2])
    await cognee.cognify()

    question = "What can I use to create a knowledge graph?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
    )
    print("\nSession ID: default_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "You sure about that?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=question
    )
    print("\nSession ID: default_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "This is awesome!"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=question
    )
    print("\nSession ID: default_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "Where is Germany?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
        session_id="different_session",
    )
    print("\nSession ID: different_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "Next to which country again?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
        session_id="different_session",
    )
    print("\nSession ID: different_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    question = "So you remember everything I asked from you?"
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
        session_id="different_session",
    )
    print("\nSession ID: different_session")
    print(f"Question: {question}")
    print(f"Answer: {search_results}\n")

    session_ids_to_persist = ["default_session", "different_session"]
    default_user = await get_default_user()

    await persist_sessions_in_knowledge_graph_pipeline(
        user=default_user,
        session_ids=session_ids_to_persist,
    )

    visualize_graph_path = os.path.join(
        os.path.dirname(__file__), ".artifacts/conversation_session_persistence.html"
    )
    await visualize_graph(visualize_graph_path)


if __name__ == "__main__":
    asyncio.run(main())

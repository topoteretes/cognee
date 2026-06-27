import asyncio
import logging

from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.GraphCompletionRetriever import GraphCompletionRetriever

logging.basicConfig(level=logging.INFO)


async def test_context_only_trace():
    print("--- Starting #2910 Feature Verification ---")

    # 1. Initialize a test session ID and retriever configuration
    test_session_id = "demo-session-2910"
    retriever = GraphCompletionRetriever(session_id=test_session_id)

    # Fake some dummy retrieved graph objects to pretend we found context match entries
    # In a real environment, these are Edge objects returned from get_retrieved_objects()
    mock_retrieved_edges = []
    mock_context_text = "Sample graph context text connecting Node A to Node B."

    print("Executing context-only retrieval with persist_trace=True...")

    # 2. Trigger our modified function passing our target flag parameters
    result = await retriever.get_completion_from_context(
        query="What connects Node A to Node B?",
        retrieved_objects=mock_retrieved_edges,
        context=mock_context_text,
        persist_trace=True,
    )

    print(f"Returned output format: {result}")

    # 3. Pull from the session storage manager history to check if it saved the QA block
    sm = get_session_manager()
    history = await sm.get_session_history(session_id=test_session_id)

    print("\n--- Session History Output Log ---")
    if history:
        for entry in history:
            print(f"Question: {getattr(entry, 'question', None)}")
            print(f"Answer Reference: {getattr(entry, 'answer', None)}")
            print(f"Persisted Graph Element IDs: {getattr(entry, 'used_graph_element_ids', None)}")
    else:
        print("Verification Failed: No session history written.")


if __name__ == "__main__":
    asyncio.run(test_context_only_trace())

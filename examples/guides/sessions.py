import asyncio
import cognee
from cognee import SearchType


async def main():
    # Start clean (optional in your app)
    await cognee.forget(everything=True)

    # Prepare knowledge base
    await cognee.remember(
        [
            "Alice moved to Paris in 2010. She works as a software engineer.",
            "Bob lives in New York. He is a data scientist.",
            "Alice and Bob met at a conference in 2015.",
        ],
        self_improvement=False,
    )

    # First recall - starts a new session (default user is used when none is passed)
    result1 = await cognee.recall(
        query_text="Where does Alice live?",
        query_type=SearchType.GRAPH_COMPLETION,
        session_id="conversation_1",
    )
    print("First answer:", result1[0].text)

    # Follow-up recall - uses conversation history
    result2 = await cognee.recall(
        query_text="What does she do for work?",
        query_type=SearchType.GRAPH_COMPLETION,
        session_id="conversation_1",  # Same session
    )
    print("Follow-up answer:", result2[0].text)
    # The LLM knows "she" refers to Alice from previous context

    # Different session - no memory of previous conversation
    result3 = await cognee.recall(
        query_text="What does she do for work?",
        query_type=SearchType.GRAPH_COMPLETION,
        session_id="conversation_2",  # New session
    )
    print("New session answer:", result3[0].text)
    # Without conversation history, the LLM cannot tell who "she" refers to


if __name__ == "__main__":
    asyncio.run(main())

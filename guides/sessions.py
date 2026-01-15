import asyncio
import cognee
from cognee import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.context_global_variables import set_session_user_context_variable

async def main():
    # Prepare knowledge base
    await cognee.add([
        "Alice moved to Paris in 2010. She works as a software engineer.",
        "Bob lives in New York. He is a data scientist.",
        "Alice and Bob met at a conference in 2015."
    ])
    await cognee.cognify()

    # Set user context (required for sessions)
    user = await get_default_user()
    await set_session_user_context_variable(user)

    # First search - starts a new session
    result1 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Where does Alice live?",
        session_id="conversation_1"
    )
    print("First answer:", result1[0])

    # Follow-up search - uses conversation history
    result2 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What does she do for work?",
        session_id="conversation_1"  # Same session
    )
    print("Follow-up answer:", result2[0])
    # The LLM knows "she" refers to Alice from previous context

    # Different session - no memory of previous conversation
    result3 = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What does she do for work?",
        session_id="conversation_2"  # New session
    )
    print("New session answer:", result3[0])
    # This won't know who "she" refers to

asyncio.run(main())

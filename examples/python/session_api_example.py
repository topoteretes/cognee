import os
import asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import setup_logging, INFO


async def main():
    if os.environ.get("CACHING") is None:
        os.environ["CACHING"] = "true"
    if os.environ.get("CACHE_BACKEND") is None:
        os.environ["CACHE_BACKEND"] = "redis"

    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Done.\n")

    text = "Cognee builds knowledge graphs from text and provides session-based feedback APIs."
    await cognee.add(text)
    await cognee.cognify()

    user = await get_default_user()

    question = "What does Cognee provide?"
    print(f"Searching: {question}")
    results = await cognee.search(
        query_text=question,
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        session_id='my_example_session'
    )
    print(f"Answer: {results}\n")

    session_id = "my_example_session"
    qas = await cognee.session.get_session(session_id=session_id, user=user)

    if qas:
        print(f"Found {len(qas)} sessions.\n")

    assert len(qas) == 1



if __name__ == "__main__":
    setup_logging(log_level=INFO)
    asyncio.run(main())

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
        session_id="my_example_session")
    print(f"Answer: {results}\n")

    question = "How are sessions related to Cognee?"
    print(f"Searching: {question}")
    results = await cognee.search(
        query_text=question,
        query_type=SearchType.GRAPH_COMPLETION,
        user=user)
    print(f"Answer: {results}\n")

    session_id = "my_example_session"
    qas = await cognee.session.get_session(session_id='my_example_session',
                                           user=user)

    default_qas = await cognee.session.get_session(session_id="default_session",
                                           user=user)
    print(f"First session {session_id!r} has {len(qas)} QA(s).")

    print(f"Search call without sesion {default_qas!r} has {len(default_qas)} QA(s).")

    latest = qas[-1]
    latest_default = default_qas[-1]
    feedback_status_1 = await cognee.session.add_feedback(
        session_id=session_id,
        qa_id=latest.qa_id,
        feedback_text="Very helpful, thanks!",
        feedback_score=5,
        user=user,
    )

    feedback_status_2 = await cognee.session.add_feedback(
        session_id='default_session',
        qa_id=latest_default.qa_id,
        feedback_text="Not that helpful, thanks!",
        feedback_score=2,
        user=user,
    )

    print(f"Non-default-feedback success: {feedback_status_1!r}.")
    print(f"default-feedback success: {feedback_status_2!r}.")

    deleted_1 = await cognee.session.delete_feedback(
        session_id=session_id,
        qa_id=latest.qa_id,
        user=user,
    )
    deleted_2 = await cognee.session.delete_feedback(
        session_id="default_session",
        qa_id=latest_default.qa_id,
        user=user,
    )
    print(f"delete_feedback (my_example_session): {deleted_1!r}.")
    print(f"delete_feedback (default_session): {deleted_2!r}.")



if __name__ == "__main__":
    setup_logging(log_level=INFO)
    asyncio.run(main())

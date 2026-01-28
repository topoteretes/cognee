import asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.memify_pipelines.feedback_enrichment_memify import feedback_enrichment_memify

CONVERSATION = [
    "Alice: Hey, Bob. Did you talk to Mallory?",
    "Bob: Yeah, I just saw her before coming here.",
    "Alice: Then she told you to bring my documents, right?",
    "Bob: Uh… not exactly. She said you wanted me to bring you donuts. Which sounded kind of odd…",
    "Alice: Ugh, she’s so annoying. Thanks for the donuts anyway!",
]


async def initialize_conversation_and_graph(conversation):
    """Prune data/system, add conversation, cognify."""
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(conversation)
    await cognee.cognify()


async def run_question_and_submit_feedback(question_text: str) -> bool:
    """Ask question, submit feedback based on correctness, and return correctness flag."""
    result = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question_text,
        save_interaction=True,
    )
    answer_text = str(result).lower()
    mentions_mallory = "mallory" in answer_text
    feedback_text = (
        "Great answers, very helpful!"
        if mentions_mallory
        else "The answer about Bob and donuts was wrong."
    )
    await cognee.search(
        query_type=SearchType.FEEDBACK,
        query_text=feedback_text,
        last_k=1,
    )
    return mentions_mallory


async def main():
    await initialize_conversation_and_graph(CONVERSATION)
    is_correct = await run_question_and_submit_feedback("Who told Bob to bring the donuts?")
    if not is_correct:
        await feedback_enrichment_memify(last_n=5)


if __name__ == "__main__":
    asyncio.run(main())

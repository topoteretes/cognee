import asyncio
import cognee

from cognee.api.v1.search import SearchType


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text = """
        Natural language processing (NLP) is an interdisciplinary
        subfield of computer science and information retrieval.
        First rule of coding: Do not talk about coding.
        """

    text2 = """
    Sandwiches are best served toasted with cheese, ham, mayo,
    lettuce, mustard, and salt & pepper.
    """

    await cognee.remember(text, dataset_name="NLP_coding", self_improvement=False)
    await cognee.remember(text2, dataset_name="Sandwiches", self_improvement=False)
    await cognee.remember(text2, self_improvement=False)

    # Make sure you've already run cognee.remember(...) so the graph has content.
    answers = await cognee.recall(
        query_text="Give me an overview of this dataset.",
        datasets=["NLP_coding"],
    )
    assert len(answers) > 0

    answers = await cognee.recall(
        query_text="Give me an overview of this dataset.",
        datasets=["Sandwiches"],
    )
    assert len(answers) > 0

    answers = await cognee.recall(
        query_text="List coding guidelines",
        query_type=SearchType.CODING_RULES,
    )
    assert len(answers) == 0

    answers = await cognee.recall(
        query_text="Give me a confident answer: What is NLP?",
        system_prompt="Answer succinctly and state confidence at the end.",
    )
    assert len(answers) > 0

    answers = await cognee.recall(query_text="Tell me about NLP", only_context=True)
    assert len(answers) > 0


if __name__ == "__main__":
    asyncio.run(main())

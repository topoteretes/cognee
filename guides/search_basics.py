import asyncio
import cognee

from cognee.modules.search.types import SearchType, CombinedSearchResult


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

    await cognee.add(text, dataset_name="NLP_coding")
    await cognee.add(text2, dataset_name="Sandwiches")
    await cognee.add(text2)

    await cognee.cognify()

    # Make sure you've already run cognee.cognify(...) so the graph has content
    answers = await cognee.search(query_text="What are the main themes in my data?")
    assert len(answers) > 0

    answers = await cognee.search(
        query_text="List coding guidelines",
        query_type=SearchType.CODING_RULES,
    )
    assert len(answers) > 0

    answers = await cognee.search(
        query_text="Give me a confident answer: What is NLP?",
        system_prompt="Answer succinctly and state confidence at the end.",
    )
    assert len(answers) > 0

    answers = await cognee.search(
        query_text="Tell me about NLP",
        only_context=True,
    )
    assert len(answers) > 0

    answers = await cognee.search(
        query_text="Quarterly financial highlights",
        datasets=["NLP_coding", "Sandwiches"],
        use_combined_context=True,
    )
    assert isinstance(answers, CombinedSearchResult)


asyncio.run(main())

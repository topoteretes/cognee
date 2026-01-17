import asyncio
import cognee


async def main():
    text = """
    In 1998 the project launched. In 2001 version 1.0 shipped. In 2004 the team merged
    with another group. In 2010 support for v1 ended.
    """

    await cognee.add(text, dataset_name="timeline_demo")

    await cognee.cognify(datasets=["timeline_demo"], temporal_cognify=True)

    from cognee.api.v1.search import SearchType

    # Before / after queries
    result = await cognee.search(
        query_type=SearchType.TEMPORAL, query_text="What happened before 2000?", top_k=10
    )

    assert result != []

    result = await cognee.search(
        query_type=SearchType.TEMPORAL, query_text="What happened after 2010?", top_k=10
    )

    assert result != []

    # Between queries
    result = await cognee.search(
        query_type=SearchType.TEMPORAL, query_text="Events between 2001 and 2004", top_k=10
    )

    assert result != []

    # Scoped descriptions
    result = await cognee.search(
        query_type=SearchType.TEMPORAL,
        query_text="Key project milestones between 1998 and 2010",
        top_k=10,
    )

    assert result != []

    result = await cognee.search(
        query_type=SearchType.TEMPORAL,
        query_text="What happened after 2004?",
        datasets=["timeline_demo"],
        top_k=10,
    )

    assert result != []


if __name__ == "__main__":
    asyncio.run(main())

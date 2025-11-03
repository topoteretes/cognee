import asyncio
import cognee


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text = "First rule of coding: Do not talk about coding."

    # Make sure you've already run cognee.cognify(...) so the graph has content
    answers = await cognee.search(query_text="What are the main themes in my data?")
    for answer in answers:
        print(answer)


asyncio.run(main())

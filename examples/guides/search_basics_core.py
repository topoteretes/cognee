import asyncio
import cognee


async def main():
    # Start clean (optional in your app)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    # Prepare knowledge base
    await cognee.add(
        [
            "Alice moved to Paris in 2010. She works as a software engineer.",
            "Bob lives in New York. He is a data scientist.",
            "Alice and Bob met at a conference in 2015.",
        ]
    )

    await cognee.cognify()

    # Make sure you've already run cognee.cognify(...) so the graph has content
    answers = await cognee.search(query_text="What are the main themes in my data?")
    for answer in answers:
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())

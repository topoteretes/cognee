import asyncio
import cognee


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

    # Make sure you've already run cognee.remember(...) so the graph has content
    answers = await cognee.recall(query_text="What are the main themes in my data?")
    for answer in answers:
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())

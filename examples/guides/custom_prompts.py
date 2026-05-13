import asyncio
import cognee
from cognee.api.v1.search import SearchType

custom_prompt = """
Extract only people and cities as entities.
Connect people to cities with the relationship "lives_in".
Ignore all other entities.
"""


async def main():
    await cognee.forget(everything=True)
    await cognee.remember(
        [
            "Alice moved to Paris in 2010, while Bob has always lived in New York.",
            "Andreas was born in Venice, but later settled in Lisbon.",
            "Diana and Tom were born and raised in Helsinki. Diana currently resides in Berlin, while Tom never moved.",
        ],
        custom_prompt=custom_prompt,
        self_improvement=False,
    )

    res = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Where does Alice live?",
    )
    print(res)


if __name__ == "__main__":
    asyncio.run(main())

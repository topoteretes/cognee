import asyncio
import cognee
from cognee.api.v1.search import SearchType

custom_prompt = """
Extract only people and cities as entities.
Connect people to cities with the relationship "lives_in".
Ignore all other entities.
"""


async def main():
    await cognee.add(
        [
            "Alice moved to Paris in 2010, while Bob has always lived in New York.",
            "Andreas was born in Venice, but later settled in Lisbon.",
            "Diana and Tom were born and raised in Helsingy. Diana currently resides in Berlin, while Tom never moved.",
        ]
    )
    await cognee.cognify(custom_prompt=custom_prompt)

    res = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Where does Alice live?",
    )
    print(res)


if __name__ == "__main__":
    asyncio.run(main())

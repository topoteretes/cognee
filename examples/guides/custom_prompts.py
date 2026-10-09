"""Override the entity-extraction prompt by passing custom_prompt to remember.

The prompt restricts extraction to people and cities linked by "lives_in"; a GRAPH_COMPLETION
recall then answers "Where does Alice live?" from the narrowed graph.

Requires: LLM_API_KEY.
Run: uv run python examples/guides/custom_prompts.py
"""

import asyncio

import cognee
from cognee import SearchType

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

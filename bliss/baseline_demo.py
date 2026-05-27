import asyncio

import cognee

from bliss_remember import get_papers


async def main() -> None:
    await cognee.forget(everything=True)
    await cognee.remember(get_papers(), self_improvement=False)

    query = "Which problem is the alpha module used for in these papers?"
    print(query)
    print(await cognee.recall(query))


if __name__ == "__main__":
    asyncio.run(main())

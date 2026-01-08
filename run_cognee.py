import asyncio
import cognee

DATASET = "sheet6_demo"


async def main():
    # optional clean slate
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add("Cognee turns documents into AI memory.", dataset_name=DATASET)

    # IMPORTANT: use datasets=[...] (not dataset_name=...)
    await cognee.cognify(datasets=[DATASET])

    results = await cognee.search(query_text="What does Cognee do?", datasets=[DATASET])

    for r in results:
        print(r.get("search_result"))


if __name__ == "__main__":
    asyncio.run(main())

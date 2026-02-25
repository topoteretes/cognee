import asyncio
import cognee
import dlt


async def main():
    # Sample data containing users and their pets
    data = [
        {
            "id": 1,
            "name": "Alice",
            "pets": [
                {"id": 1, "name": "Fluffy", "type": "cat"},
                {"id": 2, "name": "Spot", "type": "dog"},
            ],
        },
        {"id": 2, "name": "Bob", "pets": [{"id": 3, "name": "Fido", "type": "dog"}]},
        {"id": 3, "name": "Charlie", "pets": [{"id": 4, "name": "Klokan", "type": "kangaroo"}]},
    ]

    @dlt.resource()
    def users_and_pets():
        yield data

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(users_and_pets, dataset_name="users_and_pets", incremental_loading=False)

    await cognee.cognify(dlt_ingestion=True)

    from cognee.api.v1.visualize import visualize_graph

    await visualize_graph()

    result = await cognee.search("Which pet does Alice have?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

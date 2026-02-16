import asyncio
import cognee
import dlt


async def main():
    # Sample data containing pokemon details
    # data = [
    #     {"id": "1", "name": "bulbasaur", "size": {"weight": 6.9, "height": 0.7}, "strong_against": "pikachu"},
    #     {"id": "4", "name": "charmander", "size": {"weight": 8.5, "height": 0.6}, "strong_against": "bulbasaur"},
    #     {"id": "25", "name": "pikachu", "size": {"weight": 6, "height": 0.4}, "strong_against": "charmander"},
    # ]
    #
    # data_evolved = [
    #     {"id": "1", "nickname": "venusaur", "size": {"weight": 30.9, "height": 1.2}, "strong": "raichu"},
    #     {"id": "4", "nickname": "charizard", "size": {"weight": 40.5, "height": 1.6}, "strong": "venusaur"},
    #     {"id": "25", "nickname": "raichu", "size": {"weight": 14, "height": 0.8}, "strong": "charizard"},
    # ]

    # @dlt.resource(table_name="pokemon")
    # def pokemon_list():
    #     yield data
    #
    # @dlt.resource(table_name="pokemon_evolved")
    # def pokemon_list_evolved():
    #     yield data_evolved
    # @dlt.source()
    # def all_pokemon():
    #     return pokemon_list, pokemon_list_evolved

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

    await cognee.add(users_and_pets, dataset_name="users_and_pets", incremental_loading=False)

    await cognee.cognify()

    from cognee.api.v1.visualize import visualize_graph

    await visualize_graph()


if __name__ == "__main__":
    asyncio.run(main())

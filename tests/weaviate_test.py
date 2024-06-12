import asyncio

async def test_weaviate_integration():
    from cognee import config, prune, add, cognify, search

    config.set_vector_engine_provider("weaviate")
    # config.set_vector_db_url("TEST_URL")
    # config.set_vector_db_key("TEST_KEY")

    prune.prune_system()

    text = """
      Incapillo is a Pleistocene-age caldera (a depression formed by the collapse of a volcano) in the La Rioja Province of Argentina. It is the southernmost volcanic centre in the Andean Central Volcanic Zone (CVZ) that erupted during the Pleistocene. Incapillo is one of several ignimbrite[a] or caldera systems that, along with 44 active stratovolcanoes, are part of the CVZ.
      Subduction of the Nazca Plate beneath the South American Plate is responsible for most of the volcanism in the CVZ. After activity in the volcanic arc of the western Maricunga Belt ceased six million years ago, volcanism commenced in the Incapillo region, forming the high volcanic edifices Monte Pissis, Cerro Bonete Chico and Sierra de Veladero. Later, a number of lava domes were emplaced between these volcanoes.
      Incapillo is the source of the Incapillo ignimbrite, a medium-sized deposit comparable to the Katmai ignimbrite. The Incapillo ignimbrite was erupted 0.52 ± 0.03 and 0.51 ± 0.04 million years ago and has a volume of about 20.4 cubic kilometres (4.9 cu mi). A caldera with dimensions of 5 by 6 kilometres (3.1 mi × 3.7 mi) formed during the eruption. Later volcanism generated more lava domes within the caldera and a debris flow in the Sierra de Veladero. The lake within the caldera may overlie an area of ongoing hydrothermal activity.
    """

    await add(text)

    await cognify()

    result = await search("SIMILARITY", { "query": "volcanic eruption" })

    print(result)

if __name__ == "__main__":
    asyncio.run(test_weaviate_integration())

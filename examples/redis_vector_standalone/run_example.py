import asyncio
from cognee.infrastructure.databases.vector import get_cache_vector_engine
from cognee.modules.engine.models import Entity
from cognee.modules.engine.utils import generate_node_id


async def main() -> None:
    cache_engine = get_cache_vector_engine()

    collection = "cache_vector_example"

    await cache_engine.create_collection(collection)

    entities = [
        ("NLP", "Branch of AI for understanding human language."),
        ("Sandwitches", "Best when toasted with cheese, ham, mayo, lettuce."),
        ("Vector databases", "Store embeddings and support similarity search."),
    ]
    data_points = [
        Entity(id=generate_node_id(name), name=name, description=description)
        for name, description in entities
    ]
    await cache_engine.create_data_points(collection, data_points)

    query = "Sandwitches"


    results = await cache_engine.search(
        collection_name=collection,
        query_text=query,
        limit=1,
        include_payload=True,
    )
    for r in results:
        payload = r.payload or {}
        text_preview = (payload.get("name") or payload.get("text") or str(payload))[:80]
        print(f"   id={r.id} score={r.score:.4f}  name={text_preview!r}")


    precomputed = (await cache_engine.embed_data([query]))[0]
    results2 = await cache_engine.search(
        collection_name=collection,
        query_vector=precomputed,
        limit=1,
        include_payload=True,
    )
    for r in results2:
        payload = r.payload or {}
        text_preview = (payload.get("name") or payload.get("text") or str(payload))[:80]
        print(f"   id={r.id} score={r.score:.4f}  name={text_preview!r}")


if __name__ == "__main__":
    asyncio.run(main())

import os
import pathlib
import pytest

import cognee


class TestVectorEngine:
    # Test that vector engine search works well with limit=None.
    # Search should return all entities that exist in a collection. Used Alice for a bit larger test.
    @pytest.mark.asyncio
    async def test_vector_engine_search_none_limit(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_vector_engine_search_none_limit"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_vector_engine_search_none_limit"
        )
        cognee.config.data_root_directory(data_directory_path)

        file_path = os.path.join(pathlib.Path(__file__).resolve().parent, "data", "alice_in_wonderland.txt")

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        await cognee.add(file_path)

        await cognee.cognify()

        query_text = "List me all the important characters in Alice in Wonderland."

        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()

        collection_name = "Entity_name"

        query_vector = (await vector_engine.embedding_engine.embed_text([query_text]))[0]

        result = await vector_engine.search(
            collection_name=collection_name, query_vector=query_vector, limit=None
        )

        # Check that we did not accidentally use any default value for limit in vector search along the way (like 5, 10, or 15)
        assert len(result) > 15
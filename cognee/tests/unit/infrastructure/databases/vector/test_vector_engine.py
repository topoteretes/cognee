import os
import pathlib
import pytest

import cognee
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


class TestVectorEngine:
    # Test that vector engine search works well with limit=None.
    # Search should return all triplets that exist. Used Alice for a bit larger test.
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

        # Use high value to make sure we get everything that the vector search returns
        retriever = GraphCompletionRetriever(top_k=1000)

        result = await retriever.get_triplets(query_text)

        # Check that we did not accidentally use any default value for limit in vector search along the way (like 5, 10, or 15)
        assert len(result) > 15
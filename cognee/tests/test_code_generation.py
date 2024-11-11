import os
import logging
import pathlib
import cognee
from cognee.api.v1.cognify.code_graph_pipeline import code_graph_pipeline
from cognee.api.v1.search import SearchType
from cognee.shared.utils import render_graph

logging.basicConfig(level = logging.DEBUG)

async def  main():
    data_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_code_generation")).resolve())
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_code_generation")).resolve())
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata = True)

    dataset_name = "artificial_intelligence"

    ai_text_file_path = os.path.join(pathlib.Path(__file__).parent, "test_data/code.txt")
    await cognee.add([ai_text_file_path], dataset_name)

    await code_graph_pipeline([dataset_name])

    await render_graph(None, include_nodes = True, include_labels = True)

    search_results = await cognee.search(SearchType.CHUNKS, query = "Student")
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted chunks are:\n")
    for result in search_results:
        print(f"{result}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main(), debug=True)

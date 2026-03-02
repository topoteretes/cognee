import os
import asyncio
import pathlib
from cognee import config, add, cognify, search, SearchType, prune, visualize_graph

from cognee.low_level import DataPoint


# Define a custom graph model for programming languages.
class FieldType(DataPoint):
    name: str = "Field"


class Field(DataPoint):
    name: str
    is_type: FieldType
    metadata: dict = {"index_fields": ["name"]}


class ProgrammingLanguageType(DataPoint):
    name: str = "Programming Language"


class ProgrammingLanguage(DataPoint):
    name: str
    used_in: list[Field] = []
    is_type: ProgrammingLanguageType
    metadata: dict = {"index_fields": ["name"]}


def set_up_config():
    data_directory_path = str(
        pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".data_storage")).resolve()
    )
    # Set up the data directory. Cognee will store files here.
    config.data_root_directory(data_directory_path)

    cognee_directory_path = str(
        pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".cognee_system")).resolve()
    )
    # Set up the Cognee system directory. Cognee will store system files and databases here.
    config.system_root_directory(cognee_directory_path)


async def visualize_data():
    graph_file_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".artifacts/graph_visualization.html")
        ).resolve()
    )
    await visualize_graph(graph_file_path)


async def main():
    set_up_config()

    # Prune data and system metadata before running, only if we want "fresh" state.
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    text = "The Python programming language is widely used in data analysis, web development, and machine learning."

    await add(text)
    await cognify(graph_model=ProgrammingLanguage)

    await visualize_data()

    # Completion query that uses graph data to form context.
    graph_completion = await search(
        query_text="What is Python?", query_type=SearchType.GRAPH_COMPLETION
    )
    print(graph_completion)

    # Completion query that uses document chunks to form context.
    rag_completion = await search(
        query_text="What is Python?", query_type=SearchType.RAG_COMPLETION
    )
    print(rag_completion)

    # Query all summaries related to query.
    summaries = await search(query_text="Python", query_type=SearchType.SUMMARIES)
    for summary in summaries:
        print(summary)

    chunks = await search(query_text="Python", query_type=SearchType.CHUNKS)
    for chunk in chunks:
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import os

from cognee import SearchType, config, forget, recall, remember, visualize_graph
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
    data_directory_path = os.path.join(os.path.dirname(__file__), ".data_storage")
    # Set up the data directory. Cognee will store files here.
    config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(os.path.dirname(__file__), ".cognee_system")
    # Set up the Cognee system directory. Cognee will store system files and databases here.
    config.system_root_directory(cognee_directory_path)


async def visualize_data():
    graph_file_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "graph_visualization.html"
    )
    await visualize_graph(graph_file_path)


async def main():
    set_up_config()

    # Prune data and system metadata before running, only if we want "fresh" state.
    await forget(everything=True)

    text = "The Python programming language is widely used in data analysis, web development, and machine learning."

    await remember(text, graph_model=ProgrammingLanguage, self_improvement=False)

    await visualize_data()

    # Completion query that uses graph data to form context.
    graph_completion = await recall(
        query_text="What is Python?", query_type=SearchType.GRAPH_COMPLETION
    )
    print(graph_completion)

    # Completion query that uses document chunks to form context.
    rag_completion = await recall(
        query_text="What is Python?", query_type=SearchType.RAG_COMPLETION
    )
    print(rag_completion)

    # Query all summaries related to query.
    summaries = await recall(query_text="Python", query_type=SearchType.SUMMARIES)
    for summary in summaries:
        print(summary)

    chunks = await recall(query_text="Python", query_type=SearchType.CHUNKS)
    for chunk in chunks:
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())

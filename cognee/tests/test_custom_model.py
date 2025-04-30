import os
import pathlib
import cognee
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.shared.utils import render_graph
from cognee.low_level import DataPoint

logger = get_logger()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_custom_model")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_custom_model")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Define a custom graph model for programming languages.
    class FieldType(DataPoint):
        name: str = "Field"
        metadata: dict = {"index_fields": ["name"]}

    class Field(DataPoint):
        name: str
        is_type: FieldType
        metadata: dict = {"index_fields": ["name"]}

    class ProgrammingLanguageType(DataPoint):
        name: str = "Programming Language"
        metadata: dict = {"index_fields": ["name"]}

    class ProgrammingLanguage(DataPoint):
        name: str
        used_in: list[Field] = []
        is_type: ProgrammingLanguageType
        metadata: dict = {"index_fields": ["name"]}

    text = (
        "Python is an interpreted, high-level, general-purpose programming language. It was created by Guido van Rossum and first released in 1991. "
        + "Python is widely used in data analysis, web development, and machine learning."
    )

    await cognee.add(text)

    await cognee.cognify(graph_model=ProgrammingLanguage)

    url = await render_graph()
    print(f"Graphistry URL: {url}")

    graph_file_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".artifacts/test_custom_model/graph_visualization.html",
            )
        ).resolve()
    )
    await cognee.visualize_graph(graph_file_path)

    # Completion query that uses graph data to form context.
    completion = await cognee.search(SearchType.GRAPH_COMPLETION, "What is python?")
    assert len(completion) != 0, "Graph completion search didn't return any result."
    print("Graph completion result is:")
    print(completion)

    # Completion query that uses document chunks to form context.
    completion = await cognee.search(SearchType.RAG_COMPLETION, "What is Python?")
    assert len(completion) != 0, "Completion search didn't return any result."
    print("Completion result is:")
    print(completion)

    # Query all summaries related to query.
    summaries = await cognee.search(SearchType.SUMMARIES, "Python")
    assert len(summaries) != 0, "Summaries search didn't return any results."
    print("Summary results are:")
    for summary in summaries:
        print(summary)

    chunks = await cognee.search(SearchType.CHUNKS, query_text="Python")
    assert len(chunks) != 0, "Chunks search didn't return any results."
    print("Chunk results are:")
    for chunk in chunks:
        print(chunk)

    user = await get_default_user()
    history = await get_history(user.id)

    assert len(history) == 8, "Search history is not correct."


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)

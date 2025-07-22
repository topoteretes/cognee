import os
import pathlib
from uuid import uuid4

import cognee
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType

logger = get_logger()


async def main():
    bucket_name = os.getenv("STORAGE_BUCKET_NAME")
    test_run_id = uuid4()
    data_directory_path = f"s3://{bucket_name}/{test_run_id}/data"
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = f"s3://{bucket_name}/{test_run_id}/system"
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "artificial_intelligence"

    ai_text_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/artificial-intelligence.pdf"
    )
    await cognee.add([ai_text_file_path], dataset_name)

    text = """A large language model (LLM) is a language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. LLMs acquire these abilities by learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised training process. LLMs can be used for text generation, a form of generative AI, by taking an input text and repeatedly predicting the next token or word.
    LLMs are artificial neural networks. The largest and most capable, as of March 2024, are built with a decoder-only transformer-based architecture while some recent implementations are based on other architectures, such as recurrent neural network variants and Mamba (a state space model).
    Up to 2020, fine tuning was the only way a model could be adapted to be able to accomplish specific tasks. Larger sized models, such as GPT-3, however, can be prompt-engineered to achieve similar results.[6] They are thought to acquire knowledge about syntax, semantics and "ontology" inherent in human language corpora, but also inaccuracies and biases present in the corpora.
    Some notable LLMs are OpenAI's GPT series of models (e.g., GPT-3.5 and GPT-4, used in ChatGPT and Microsoft Copilot), Google's PaLM and Gemini (the latter of which is currently used in the chatbot of the same name), xAI's Grok, Meta's LLaMA family of open-source models, Anthropic's Claude models, Mistral AI's open source models, and Databricks' open source DBRX.
    """

    await cognee.add([text], dataset_name)

    await cognee.cognify([dataset_name])

    from cognee.infrastructure.databases.vector import get_vector_engine

    vector_engine = get_vector_engine()
    random_node = (await vector_engine.search("Entity_name", "AI"))[0]
    random_node_name = random_node.payload["text"]

    search_results = await cognee.search(
        query_type=SearchType.INSIGHTS, query_text=random_node_name
    )
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(query_type=SearchType.CHUNKS, query_text=random_node_name)
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted chunks are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(
        query_type=SearchType.SUMMARIES, query_text=random_node_name
    )
    assert len(search_results) != 0, "Query related summaries don't exist."
    print("\nExtracted summaries are:\n")
    for result in search_results:
        print(f"{result}\n")

    user = await get_default_user()
    history = await get_history(user.id)

    assert len(history) == 6, "Search history is not correct."

    # Assert local data files are cleaned properly
    await cognee.prune.prune_data()
    data_root_directory = get_storage_config()["data_root_directory"]
    assert not os.path.isdir(data_root_directory), "Local data files are not deleted"

    # Assert relational, vector and graph databases have been cleaned properly
    await cognee.prune.prune_system(metadata=True)

    connection = await vector_engine.get_connection()
    collection_names = await connection.table_names()
    assert len(collection_names) == 0, "LanceDB vector database is not empty"

    from cognee.infrastructure.databases.relational import get_relational_engine

    db_path = get_relational_engine().db_path
    dir_path = os.path.dirname(db_path)
    file_name = os.path.basename(db_path)
    file_storage = get_file_storage(dir_path)

    assert not await file_storage.file_exists(file_name), (
        "SQLite relational database is not deleted"
    )

    from cognee.infrastructure.databases.graph import get_graph_config

    graph_config = get_graph_config()
    # For Kuzu v0.11.0+, check if database file doesn't exist (single-file format with .kuzu extension)
    # For older versions or other providers, check if directory is empty
    if graph_config.graph_database_provider.lower() == "kuzu":
        assert not os.path.exists(graph_config.graph_file_path), (
            "Kuzu graph database file still exists"
        )
    else:
        assert not os.path.exists(graph_config.graph_file_path) or not os.listdir(
            graph_config.graph_file_path
        ), "Graph database directory is not empty"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)

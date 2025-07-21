import os
import cognee
import pathlib
from cognee.infrastructure.files.storage import get_storage_config
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType

logger = get_logger()


async def check_falkordb_connection():
    """Check if FalkorDB is available at localhost:6379"""
    try:
        from falkordb import FalkorDB

        client = FalkorDB(host="localhost", port=6379)
        # Try to list graphs to check connection
        client.list_graphs()
        return True
    except Exception as e:
        logger.warning(f"FalkorDB not available at localhost:6379: {e}")
        return False


async def main():
    # Check if FalkorDB is available
    if not await check_falkordb_connection():
        print("⚠️  FalkorDB is not available at localhost:6379")
        print("   To run this test, start FalkorDB server:")
        print("   docker run -p 6379:6379 falkordb/falkordb:latest")
        print("   Skipping FalkorDB test...")
        return

    print("✅ FalkorDB connection successful, running test...")

    # Configure FalkorDB as the graph database provider
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": "localhost",  # FalkorDB URL (using Redis protocol)
            "graph_database_port": 6379,
            "graph_database_provider": "falkordb",
        }
    )

    # Configure FalkorDB as the vector database provider too since it's a hybrid adapter
    cognee.config.set_vector_db_config(
        {
            "vector_db_url": "localhost",
            "vector_db_port": 6379,
            "vector_db_provider": "falkordb",
        }
    )

    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_falkordb")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_falkordb")
        ).resolve()
    )
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
    random_node = (await vector_engine.search("entity.name", "AI"))[0]
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

    # For FalkorDB vector engine, check if collections are empty
    # Since FalkorDB is a hybrid adapter, we can check if the graph is empty
    # as the vector data is stored in the same graph
    if hasattr(vector_engine, "driver"):
        # This is FalkorDB - check if graphs exist
        collections = vector_engine.driver.list_graphs()
        # The graph should be deleted, so either no graphs or empty graph
        if vector_engine.graph_name in collections:
            # Graph exists but should be empty
            vector_graph_data = await vector_engine.get_graph_data()
            vector_nodes, vector_edges = vector_graph_data
            assert len(vector_nodes) == 0 and len(vector_edges) == 0, (
                "FalkorDB vector database is not empty"
            )
    else:
        # Fallback for other vector engines like LanceDB
        connection = await vector_engine.get_connection()
        collection_names = await connection.table_names()
        assert len(collection_names) == 0, "Vector database is not empty"

    from cognee.infrastructure.databases.relational import get_relational_engine

    assert not os.path.exists(get_relational_engine().db_path), (
        "SQLite relational database is not empty"
    )

    # For FalkorDB, check if the graph database is empty
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = get_graph_engine()
    graph_data = await graph_engine.get_graph_data()
    nodes, edges = graph_data
    assert len(nodes) == 0 and len(edges) == 0, "FalkorDB graph database is not empty"

    print("🎉 FalkorDB test completed successfully!")
    print("   ✓ Data ingestion worked")
    print("   ✓ Cognify processing worked")
    print("   ✓ Search operations worked")
    print("   ✓ Cleanup worked")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)

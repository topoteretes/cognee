import base64
import os
import pathlib
from uuid import UUID

import cognee
from cognee import update
from cognee.infrastructure.files.encryption.envelope import MAGIC
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.search.operations import get_history
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.tests.utils.get_latest_pipeline_run import get_latest_pipeline_run

logger = get_logger()


async def assert_documents_sealed_and_readable(dataset_name: str, expected_texts: dict[UUID, str]):
    """Every object under the data root is an envelope on disk, every data item of
    the dataset reads back as plaintext through the storage layer, and the items
    in ``expected_texts`` (by data id) read back as exactly that text — the
    document that was processed, not what encryption made of it.
    """
    data_root = get_storage_config()["data_root_directory"]
    stored_files = [
        os.path.join(directory, name)
        for directory, _, names in os.walk(data_root)
        for name in names
    ]
    assert stored_files, "No stored objects under the data root."
    for path in stored_files:
        with open(path, "rb") as stored:
            assert stored.read(len(MAGIC)) == MAGIC, f"{path} is stored in plaintext."

    user = await get_default_user()
    dataset = (await get_datasets_by_name([dataset_name], user.id))[0]
    data_items = await get_dataset_data(dataset.id)
    assert data_items, "The dataset has no data items to check."
    read_back = {}
    for item in data_items:
        with open(get_data_file_path(item.raw_data_location), "rb") as stored:
            assert stored.read(len(MAGIC)) == MAGIC, f"{item.name} is stored in plaintext."
        async with open_data_file(item.raw_data_location, mode="rb") as plaintext:
            content = plaintext.read()
        assert not content.startswith(MAGIC), f"{item.name} read back as ciphertext."
        read_back[item.id] = content.decode("utf-8")

    for data_id, expected in expected_texts.items():
        assert data_id in read_back, f"Data item {data_id} is missing from the dataset."
        assert read_back[data_id] == expected, (
            f"Data item {data_id} read back differs from the document that was processed."
        )


async def main():
    # Disable session-turn gating (auto_feedback). This script issues several searches against the
    # same session; with gating on, the turn analysis can intercept a follow-up search with a
    # clarifying acknowledgement instead of an answer. That layer has dedicated coverage
    # (e.g. test_session_context_turn_flow.py); here we assert direct retrieval results.
    os.environ["AUTO_FEEDBACK"] = "False"

    # Run the whole pipeline with file encryption on: every object under the data
    # root must be sealed on disk, and add/cognify/search/update must read the
    # same documents back through the layer as they would in plaintext.
    os.environ["FILE_ENCRYPTION_ENABLED"] = "true"
    os.environ["FILE_ENCRYPTION_KEY"] = base64.b64encode(os.urandom(32)).decode()

    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_library")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_library")
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

    cognify_run_info = await cognee.cognify([dataset_name])

    cognify_run = await get_latest_pipeline_run(
        pipeline_name="cognify_pipeline", status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED
    )
    assert cognify_run.embedding_tokens > 0, "Cognify recorded no embedding tokens."

    # The processed documents are sealed on disk and the text item (the dataset
    # holds the PDF and the text at this point) reads back as exactly what was added.
    user = await get_default_user()
    dataset = (await get_datasets_by_name([dataset_name], user.id))[0]
    text_item = next(
        item for item in await get_dataset_data(dataset.id) if item.name.startswith("text_")
    )
    await assert_documents_sealed_and_readable(dataset_name, {text_item.id: text})

    from cognee.infrastructure.databases.vector import get_vector_engine_async

    vector_engine = await get_vector_engine_async()
    random_node = (await vector_engine.search("Entity_name", "AI", include_payload=True))[0]
    random_node_name = random_node.payload["text"]

    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=random_node_name
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

    memory_text = (
        "Grace Hopper designed one of the first compilers and popularized the term "
        "debugging in computing."
    )
    remember_result = await cognee.remember(
        memory_text,
        dataset_name=dataset_name,
        self_improvement=False,
    )
    assert remember_result.status == "completed", "Remember did not complete successfully."
    assert remember_result.dataset_name == dataset_name, "Remember used the wrong dataset."

    recall_results = await cognee.recall(
        query_text="Grace Hopper compiler debugging",
        query_type=SearchType.CHUNKS,
        datasets=[dataset_name],
        top_k=5,
        auto_route=False,
    )
    assert len(recall_results) != 0, "Recall results list is empty."
    recall_text = "\n".join(result.text for result in recall_results).lower()
    assert "grace hopper" in recall_text, "Recall did not return remembered content."
    assert "compiler" in recall_text, "Recall result is missing the remembered compiler detail."

    # A CHUNKS recall only embeds the query: its tokens are embedding tokens, not LLM ones.
    recall_run = await get_latest_pipeline_run(operation_name="recall")
    assert recall_run.embedding_tokens > 0, "Recall recorded no embedding tokens."
    assert recall_run.tokens_in == 0, "Recall's embedding tokens were counted as LLM tokens."

    # Test updating of documents
    # Get Pipeline Run object
    pipeline_run_obj = next(iter(cognify_run_info.values()))
    for data_item in pipeline_run_obj.data_ingestion_info:
        # Update all documents in dataset to only contain Mark and Cindy information
        await update(
            dataset_id=pipeline_run_obj.dataset_id,
            data_id=data_item["data_id"],
            data="Mark met with Cindy at a cafe.",
        )

    # The rewritten documents are sealed too and read back as the new content.
    await assert_documents_sealed_and_readable(
        dataset_name,
        {
            UUID(str(data_item["data_id"])): "Mark met with Cindy at a cafe."
            for data_item in pipeline_run_obj.data_ingestion_info
        },
    )

    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What information do you contain?",
        dataset_ids=[pipeline_run_obj.dataset_id],
    )
    result_text = search_results[0]["search_result"][0].lower()
    assert "mark" in result_text, "Failed to update document, no mention of Mark in search results"
    assert "cindy" in result_text, (
        "Failed to update document, no mention of Cindy in search results"
    )
    assert "artificial intelligence" not in result_text, (
        "Failed to update document, Artificial intelligence still mentioned in search results"
    )

    # Test visualization
    from cognee import visualize_graph

    await visualize_graph(dataset="artificial_intelligence")

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
    file_path = os.path.basename(db_path)
    file_storage = get_file_storage(dir_path)

    assert not await file_storage.file_exists(file_path), (
        "SQLite relational database is not deleted"
    )

    from cognee.infrastructure.databases.graph import get_graph_config

    graph_config = get_graph_config()
    # For Ladybug/Kuzu, check if database file doesn't exist.
    # For older versions or other providers, check if directory is empty
    if graph_config.graph_database_provider.lower() in ("ladybug", "kuzu"):
        assert not os.path.exists(graph_config.graph_file_path), (
            "Ladybug graph database file still exists"
        )
    else:
        assert not os.path.exists(graph_config.graph_file_path) or not os.listdir(
            graph_config.graph_file_path
        ), "Graph database directory is not empty"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)

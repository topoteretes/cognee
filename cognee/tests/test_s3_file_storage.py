import os
import pathlib
from io import BytesIO
from uuid import uuid4

from fastapi import UploadFile

import cognee
from cognee.api.v1.ontologies.ontologies import OntologyService
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.search.operations import get_history
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()

ONTOLOGY_KEY = "s3_storage_test_ontology"
ONTOLOGY_PATH = (
    pathlib.Path(__file__).parents[2] / "examples/guides/ontology_input_example/basic_ontology.owl"
)


async def list_user_ontology_objects(data_directory_path: str, user) -> list[str]:
    """Objects under the user's ontology folder, read from the bucket itself.

    An ``s3://`` path always resolves to ``S3FileStorage``, whatever the code under
    test does, so this is an S3 listing and not a look at a local folder.
    """
    return await get_file_storage(data_directory_path).list_files(str(user.id))


async def upload_ontology_and_read_it_back(data_directory_path: str, user) -> list:
    """Upload an ontology, check it is stored on S3 and return it as cognify reads it."""
    ontology_content = ONTOLOGY_PATH.read_bytes()
    ontology_service = OntologyService()

    await ontology_service.upload_ontology(
        ontology_key=ONTOLOGY_KEY,
        file=UploadFile(file=BytesIO(ontology_content), filename=ONTOLOGY_PATH.name),
        user=user,
    )

    # Read as a local path, the s3:// data root is a folder named "s3:" in the working
    # directory. Nothing of the upload may land there.
    local_ontology_folder = pathlib.Path(os.path.abspath(data_directory_path)) / str(user.id)
    assert not local_ontology_folder.exists(), (
        f"The ontology upload wrote to a local folder instead of S3: {local_ontology_folder}"
    )

    stored_objects = await list_user_ontology_objects(data_directory_path, user)
    assert f"{user.id}/metadata.json" in stored_objects, (
        f"The ontology index is not an object under the S3 data root: {stored_objects}"
    )
    ontology_objects = [path for path in stored_objects if path.endswith(".owl")]
    assert len(ontology_objects) == 1, (
        f"The ontology file is not an object under the S3 data root: {stored_objects}"
    )
    async with get_file_storage(data_directory_path).open(ontology_objects[0], "rb") as file:
        assert file.read() == ontology_content, "The ontology object on S3 differs from the upload"

    ontology_files = await ontology_service.get_ontology_files([ONTOLOGY_KEY], user)
    assert len(ontology_files) == 1, "The uploaded ontology was not read back from S3"
    assert ontology_files[0].read() == ontology_content, (
        "The ontology read back from S3 differs from the upload"
    )
    ontology_files[0].seek(0)
    assert ONTOLOGY_KEY in await ontology_service.list_ontologies(user), (
        "The uploaded ontology is not listed"
    )
    assert await ontology_service.export_ontology(ONTOLOGY_KEY, user), (
        "The ontology stored on S3 could not be exported"
    )

    return ontology_files


async def delete_ontology_and_check_it_is_gone(data_directory_path: str, user) -> None:
    ontology_service = OntologyService()

    await ontology_service.delete_ontology(ONTOLOGY_KEY, user)

    stored_objects = await list_user_ontology_objects(data_directory_path, user)
    assert not [path for path in stored_objects if path.endswith(".owl")], (
        f"The deleted ontology is still an object on S3: {stored_objects}"
    )
    assert ONTOLOGY_KEY not in await ontology_service.list_ontologies(user), (
        "The deleted ontology is still listed"
    )


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

    # An uploaded ontology must live in the bucket like every other upload, and cognify
    # must be able to use it from there.
    user = await get_default_user()
    ontology_files = await upload_ontology_and_read_it_back(data_directory_path, user)

    await cognee.cognify(
        [dataset_name],
        config={
            "ontology_config": {
                "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_files)
            }
        },
    )

    await delete_ontology_and_check_it_is_gone(data_directory_path, user)

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
    # For Ladybug/Kuzu, check if database file doesn't exist.
    # For older versions or other providers, check if directory is empty
    if graph_config.graph_database_provider.lower() in ("ladybug", "kuzu"):
        graph_dir = os.path.dirname(graph_config.graph_file_path)
        graph_file = os.path.basename(graph_config.graph_file_path)
        graph_storage = get_file_storage(graph_dir)
        assert not await graph_storage.file_exists(graph_file), (
            "Ladybug graph database file still exists"
        )
    else:
        graph_storage = get_file_storage(graph_config.graph_file_path)
        graph_exists = await graph_storage.file_exists("")
        graph_is_file = await graph_storage.is_file("")
        graph_files = await graph_storage.list_files("", recursive=True)
        assert not graph_exists or (not graph_is_file and not graph_files), (
            "Graph database directory is not empty"
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)

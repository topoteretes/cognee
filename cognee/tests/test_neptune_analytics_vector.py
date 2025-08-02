import os
import pathlib
import cognee
import uuid
import pytest
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
    NeptuneAnalyticsAdapter,
    IndexSchema,
)

logger = get_logger()


async def main():
    graph_id = os.getenv("GRAPH_ID", "")
    cognee.config.set_vector_db_provider("neptune_analytics")
    cognee.config.set_vector_db_url(f"neptune-graph://{graph_id}")
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_neptune")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_neptune")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "cs_explanations"

    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )
    await cognee.add([explanation_file_path], dataset_name)

    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the preparation and manipulation of quantum states.
    Classical physics cannot explain the operation of these quantum devices, and a scalable quantum computer could perform some calculations exponentially faster (with respect to input size scaling) than any modern "classical" computer. In particular, a large-scale quantum computer could break widely used encryption schemes and aid physicists in performing physical simulations; however, the current state of the technology is largely experimental and impractical, with several obstacles to useful applications. Moreover, scalable quantum computers do not hold promise for many practical tasks, and for many important tasks quantum speedups are proven impossible.
    The basic unit of information in quantum computing is the qubit, similar to the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis" states. When measuring a qubit, the result is a probabilistic output of a classical bit, therefore making quantum computers nondeterministic in general. If a quantum computer manipulates the qubit in a particular way, wave interference effects can amplify the desired measurement results. The design of quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently and quickly.
    Physically engineering high-quality qubits has proven challenging. If a physical qubit is not sufficiently isolated from its environment, it suffers from quantum decoherence, introducing noise into calculations. Paradoxically, perfectly isolating qubits is also undesirable because quantum computations typically need to initialize qubits, perform controlled qubit interactions, and measure the resulting quantum states. Each of those operations introduces errors and suffers from noise, and such inaccuracies accumulate.
    In principle, a non-quantum (classical) computer can solve the same computational problems as a quantum computer, given enough time. Quantum advantage comes in the form of time complexity rather than computability, and quantum complexity theory shows that some quantum algorithms for carefully selected tasks require exponentially fewer computational steps than the best known non-quantum algorithms. Such tasks can in theory be solved on a large-scale quantum computer whereas classical computers would not finish computations in any reasonable amount of time. However, quantum speedup is not universal or even typical across computational tasks, since basic tasks such as sorting are proven to not allow any asymptotic quantum speedup. Claims of quantum supremacy have drawn significant attention to the discipline, but are demonstrated on contrived tasks, while near-term practical use cases remain limited.
    """

    await cognee.add([text], dataset_name)

    await cognee.cognify([dataset_name])

    vector_engine = get_vector_engine()
    random_node = (await vector_engine.search("Entity_name", "Quantum computer"))[0]
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

    await cognee.prune.prune_data()
    assert not os.path.isdir(data_directory_path), "Local data files are not deleted"

    await cognee.prune.prune_system(metadata=True)


async def vector_backend_api_test():
    cognee.config.set_vector_db_provider("neptune_analytics")

    # When URL is absent
    cognee.config.set_vector_db_url(None)
    with pytest.raises(OSError):
        get_vector_engine()

    # Assert invalid graph ID.
    cognee.config.set_vector_db_url("invalid_url")
    with pytest.raises(ValueError):
        get_vector_engine()

    # Return a valid engine object with valid URL.
    graph_id = os.getenv("GRAPH_ID", "")
    cognee.config.set_vector_db_url(f"neptune-graph://{graph_id}")
    engine = get_vector_engine()
    assert isinstance(engine, NeptuneAnalyticsAdapter)

    TEST_COLLECTION_NAME = "test"
    # Data point - 1
    TEST_UUID = str(uuid.uuid4())
    TEST_TEXT = "Hello world"
    datapoint = IndexSchema(id=TEST_UUID, text=TEST_TEXT)
    # Data point - 2
    TEST_UUID_2 = str(uuid.uuid4())
    TEST_TEXT_2 = "Cognee"
    datapoint_2 = IndexSchema(id=TEST_UUID_2, text=TEST_TEXT_2)

    # Prun all vector_db entries
    await engine.prune()

    # Always return true
    has_collection = await engine.has_collection(TEST_COLLECTION_NAME)
    assert has_collection
    # No-op
    await engine.create_collection(TEST_COLLECTION_NAME, IndexSchema)

    # Save data-points
    await engine.create_data_points(TEST_COLLECTION_NAME, [datapoint, datapoint_2])
    # Search single text
    result_search = await engine.search(
        collection_name=TEST_COLLECTION_NAME,
        query_text=TEST_TEXT,
        query_vector=None,
        limit=10,
        with_vector=True,
    )
    assert len(result_search) == 2

    # # Retrieve data-points
    result = await engine.retrieve(TEST_COLLECTION_NAME, [TEST_UUID, TEST_UUID_2])
    assert any(str(r.id) == TEST_UUID and r.payload["text"] == TEST_TEXT for r in result)
    assert any(str(r.id) == TEST_UUID_2 and r.payload["text"] == TEST_TEXT_2 for r in result)
    # Search multiple
    result_search_batch = await engine.batch_search(
        collection_name=TEST_COLLECTION_NAME,
        query_texts=[TEST_TEXT, TEST_TEXT_2],
        limit=10,
        with_vectors=False,
    )
    assert len(result_search_batch) == 2 and all(len(batch) == 2 for batch in result_search_batch)

    # Delete datapoint from vector store
    await engine.delete_data_points(TEST_COLLECTION_NAME, [TEST_UUID, TEST_UUID_2])

    # Retrieve should return an empty list.
    result_deleted = await engine.retrieve(TEST_COLLECTION_NAME, [TEST_UUID])
    assert result_deleted == []


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    asyncio.run(vector_backend_api_test())

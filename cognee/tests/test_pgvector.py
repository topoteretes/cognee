import os
import pathlib
import cognee
from cognee.modules.search.operations import get_history
from cognee.shared.logging_utils import get_logger
from cognee.modules.data.models import Data
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user

logger = get_logger()


async def test_local_file_deletion(data_text, file_location):
    from sqlalchemy import select
    import hashlib
    from cognee.infrastructure.databases.relational import get_relational_engine

    engine = get_relational_engine()

    async with engine.get_async_session() as session:
        # Get hash of data contents
        encoded_text = data_text.encode("utf-8")
        data_hash = hashlib.md5(encoded_text).hexdigest()
        # Get data entry from database based on hash contents
        data = (await session.scalars(select(Data).where(Data.content_hash == data_hash))).one()
        assert os.path.isfile(data.raw_data_location), (
            f"Data location doesn't exist: {data.raw_data_location}"
        )
        # Test deletion of data along with local files created by cognee
        await engine.delete_data_entity(data.id)
        assert not os.path.exists(data.raw_data_location), (
            f"Data location still exists after deletion: {data.raw_data_location}"
        )

    async with engine.get_async_session() as session:
        # Get data entry from database based on file path
        data = (
            await session.scalars(select(Data).where(Data.raw_data_location == file_location))
        ).one()
        assert os.path.isfile(data.raw_data_location), (
            f"Data location doesn't exist: {data.raw_data_location}"
        )
        # Test local files not created by cognee won't get deleted
        await engine.delete_data_entity(data.id)
        assert os.path.exists(data.raw_data_location), (
            f"Data location doesn't exists: {data.raw_data_location}"
        )


async def test_getting_of_documents(dataset_name_1):
    # Test getting of documents for search per dataset
    from cognee.modules.users.permissions.methods import get_document_ids_for_user

    user = await get_default_user()
    document_ids = await get_document_ids_for_user(user.id, [dataset_name_1])
    assert len(document_ids) == 1, (
        f"Number of expected documents doesn't match {len(document_ids)} != 1"
    )

    # Test getting of documents for search when no dataset is provided
    user = await get_default_user()
    document_ids = await get_document_ids_for_user(user.id)
    assert len(document_ids) == 2, (
        f"Number of expected documents doesn't match {len(document_ids)} != 2"
    )


async def main():
    cognee.config.set_vector_db_config(
        {"vector_db_url": "", "vector_db_key": "", "vector_db_provider": "pgvector"}
    )
    cognee.config.set_relational_db_config(
        {
            "db_path": "",
            "db_name": "cognee_db",
            "db_host": "127.0.0.1",
            "db_port": "5432",
            "db_username": "cognee",
            "db_password": "cognee",
            "db_provider": "postgres",
        }
    )

    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_pgvector")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_pgvector")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name_1 = "natural_language"
    dataset_name_2 = "quantum"

    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )
    await cognee.add([explanation_file_path], dataset_name_1)

    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the preparation and manipulation of quantum states.
    Classical physics cannot explain the operation of these quantum devices, and a scalable quantum computer could perform some calculations exponentially faster (with respect to input size scaling) than any modern "classical" computer. In particular, a large-scale quantum computer could break widely used encryption schemes and aid physicists in performing physical simulations; however, the current state of the technology is largely experimental and impractical, with several obstacles to useful applications. Moreover, scalable quantum computers do not hold promise for many practical tasks, and for many important tasks quantum speedups are proven impossible.
    The basic unit of information in quantum computing is the qubit, similar to the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis" states. When measuring a qubit, the result is a probabilistic output of a classical bit, therefore making quantum computers nondeterministic in general. If a quantum computer manipulates the qubit in a particular way, wave interference effects can amplify the desired measurement results. The design of quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently and quickly.
    Physically engineering high-quality qubits has proven challenging. If a physical qubit is not sufficiently isolated from its environment, it suffers from quantum decoherence, introducing noise into calculations. Paradoxically, perfectly isolating qubits is also undesirable because quantum computations typically need to initialize qubits, perform controlled qubit interactions, and measure the resulting quantum states. Each of those operations introduces errors and suffers from noise, and such inaccuracies accumulate.
    In principle, a non-quantum (classical) computer can solve the same computational problems as a quantum computer, given enough time. Quantum advantage comes in the form of time complexity rather than computability, and quantum complexity theory shows that some quantum algorithms for carefully selected tasks require exponentially fewer computational steps than the best known non-quantum algorithms. Such tasks can in theory be solved on a large-scale quantum computer whereas classical computers would not finish computations in any reasonable amount of time. However, quantum speedup is not universal or even typical across computational tasks, since basic tasks such as sorting are proven to not allow any asymptotic quantum speedup. Claims of quantum supremacy have drawn significant attention to the discipline, but are demonstrated on contrived tasks, while near-term practical use cases remain limited.
    """

    await cognee.add([text], dataset_name_2)

    await cognee.cognify([dataset_name_2, dataset_name_1])

    from cognee.infrastructure.databases.vector import get_vector_engine

    await test_getting_of_documents(dataset_name_1)

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

    search_results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text=random_node_name, datasets=[dataset_name_2]
    )
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted chunks are:\n")
    for result in search_results:
        print(f"{result}\n")

    graph_completion = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=random_node_name,
        datasets=[dataset_name_2],
    )
    assert len(graph_completion) != 0, "Completion result is empty."
    print("Completion result is:")
    print(graph_completion)

    search_results = await cognee.search(
        query_type=SearchType.SUMMARIES, query_text=random_node_name
    )
    assert len(search_results) != 0, "Query related summaries don't exist."
    print("\n\nExtracted summaries are:\n")
    for result in search_results:
        print(f"{result}\n")

    user = await get_default_user()
    history = await get_history(user.id)
    assert len(history) == 8, "Search history is not correct."

    await test_local_file_deletion(text, explanation_file_path)

    await cognee.prune.prune_data()
    assert not os.path.isdir(data_directory_path), "Local data files are not deleted"

    await cognee.prune.prune_system(metadata=True)
    tables_in_database = await vector_engine.get_table_names()
    assert len(tables_in_database) == 0, "PostgreSQL database is not empty"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

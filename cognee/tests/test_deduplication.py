import hashlib
import os
from cognee.shared.logging_utils import get_logger
import pathlib

import cognee
from cognee.infrastructure.databases.relational import get_relational_engine

logger = get_logger()


async def test_deduplication():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    relational_engine = get_relational_engine()

    dataset_name = "test_deduplication"
    dataset_name2 = "test_deduplication2"

    # Test deduplication of local files
    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )
    explanation_file_path2 = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing_copy.txt"
    )
    await cognee.add([explanation_file_path], dataset_name)
    await cognee.add([explanation_file_path2], dataset_name2)

    result = await relational_engine.get_all_data_from_table("data")
    assert len(result) == 1, "More than one data entity was found."
    assert result[0]["name"] == "Natural_language_processing_copy", (
        "Result name does not match expected value."
    )

    result = await relational_engine.get_all_data_from_table("datasets")
    assert len(result) == 2, "Unexpected number of datasets found."
    assert result[0]["name"] == dataset_name, "Result name does not match expected value."
    assert result[1]["name"] == dataset_name2, "Result name does not match expected value."

    result = await relational_engine.get_all_data_from_table("dataset_data")
    assert len(result) == 2, "Unexpected number of dataset data relationships found."
    assert result[0]["data_id"] == result[1]["data_id"], "Data item is not reused between datasets."
    assert result[0]["dataset_id"] != result[1]["dataset_id"], "Dataset items are not different."

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Test deduplication of text input
    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
        At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the preparation and manipulation of quantum states.
        Classical physics cannot explain the operation of these quantum devices, and a scalable quantum computer could perform some calculations exponentially faster (with respect to input size scaling) than any modern "classical" computer. In particular, a large-scale quantum computer could break widely used encryption schemes and aid physicists in performing physical simulations; however, the current state of the technology is largely experimental and impractical, with several obstacles to useful applications. Moreover, scalable quantum computers do not hold promise for many practical tasks, and for many important tasks quantum speedups are proven impossible.
        The basic unit of information in quantum computing is the qubit, similar to the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis" states. When measuring a qubit, the result is a probabilistic output of a classical bit, therefore making quantum computers nondeterministic in general. If a quantum computer manipulates the qubit in a particular way, wave interference effects can amplify the desired measurement results. The design of quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently and quickly.
        Physically engineering high-quality qubits has proven challenging. If a physical qubit is not sufficiently isolated from its environment, it suffers from quantum decoherence, introducing noise into calculations. Paradoxically, perfectly isolating qubits is also undesirable because quantum computations typically need to initialize qubits, perform controlled qubit interactions, and measure the resulting quantum states. Each of those operations introduces errors and suffers from noise, and such inaccuracies accumulate.
        In principle, a non-quantum (classical) computer can solve the same computational problems as a quantum computer, given enough time. Quantum advantage comes in the form of time complexity rather than computability, and quantum complexity theory shows that some quantum algorithms for carefully selected tasks require exponentially fewer computational steps than the best known non-quantum algorithms. Such tasks can in theory be solved on a large-scale quantum computer whereas classical computers would not finish computations in any reasonable amount of time. However, quantum speedup is not universal or even typical across computational tasks, since basic tasks such as sorting are proven to not allow any asymptotic quantum speedup. Claims of quantum supremacy have drawn significant attention to the discipline, but are demonstrated on contrived tasks, while near-term practical use cases remain limited.
        """

    await cognee.add([text], dataset_name)
    await cognee.add([text], dataset_name2)

    result = await relational_engine.get_all_data_from_table("data")
    assert len(result) == 1, "More than one data entity was found."
    assert hashlib.md5(text.encode("utf-8")).hexdigest() in result[0]["name"], (
        "Content hash is not a part of file name."
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Test deduplication of image files
    explanation_file_path = os.path.join(pathlib.Path(__file__).parent, "test_data/example.png")
    explanation_file_path2 = os.path.join(
        pathlib.Path(__file__).parent, "test_data/example_copy.png"
    )

    await cognee.add([explanation_file_path], dataset_name)
    await cognee.add([explanation_file_path2], dataset_name2)

    result = await relational_engine.get_all_data_from_table("data")
    assert len(result) == 1, "More than one data entity was found."

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Test deduplication of sound files
    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/text_to_speech.mp3"
    )
    explanation_file_path2 = os.path.join(
        pathlib.Path(__file__).parent, "test_data/text_to_speech_copy.mp3"
    )

    await cognee.add([explanation_file_path], dataset_name)
    await cognee.add([explanation_file_path2], dataset_name2)

    result = await relational_engine.get_all_data_from_table("data")
    assert len(result) == 1, "More than one data entity was found."

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def test_deduplication_postgres():
    cognee.config.set_vector_db_config(
        {"vector_db_url": "", "vector_db_key": "", "vector_db_provider": "pgvector"}
    )
    cognee.config.set_relational_db_config(
        {
            "db_name": "cognee_db",
            "db_host": "127.0.0.1",
            "db_port": "5432",
            "db_username": "cognee",
            "db_password": "cognee",
            "db_provider": "postgres",
        }
    )

    await test_deduplication()


async def test_deduplication_sqlite():
    cognee.config.set_vector_db_config(
        {"vector_db_url": "", "vector_db_key": "", "vector_db_provider": "lancedb"}
    )
    cognee.config.set_relational_db_config(
        {
            "db_provider": "sqlite",
        }
    )

    await test_deduplication()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_deduplication")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_deduplication")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await test_deduplication_postgres()
    await test_deduplication_sqlite()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

import os
import logging
import pathlib
import cognee

logging.basicConfig(level=logging.DEBUG)

async def main():
    cognee.config.set_vector_engine_provider("weaviate")
    data_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_weaviate")).resolve())
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_weaviate")).resolve())
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata = True)

    dataset_name = "cs_explanations"

    explanation_file_path = os.path.join(pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt")
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

    from cognee.infrastructure.databases.vector import get_vector_engine
    vector_engine = get_vector_engine()
    random_node = (await vector_engine.search("entities", "AI"))[0]
    random_node_name = random_node.payload["name"]

    search_results = await cognee.search("SIMILARITY", { "query": random_node_name })
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search("TRAVERSE", { "query": random_node_name })
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search("SUMMARY", { "query": random_node_name })
    assert len(search_results) != 0, "Query related summaries don't exist."
    print("\n\nQuery related summaries exist:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search("ADJACENT", { "query": random_node_name })
    assert len(search_results) != 0, "Large language model query found no neighbours."
    print("\n\Large language model query found neighbours.\n")
    for result in search_results:
        print(f"{result}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# Get the current directory's path
from os import listdir, path
current_dir = path.abspath(".")

# Get the parent directory (one level above)
parent_dir = path.dirname(current_dir)


from cognee import config, add, cognify, search
from cognee.utils import render_graph
from os import listdir, path

data_directory_path = path.abspath("../.data")

print(data_directory_path)

config.data_path(data_directory_path)

# dataset_name = "pravilnik.energetska efikasnost.sertifikati"
# await add("file://" + path.abspath("../.test_data/062c22df-d99b-599f-90cd-2d325c8bcf69.txt"), dataset_name)


async def  main():

    dataset_name = "izmene"
    await add("data://" + path.abspath("../.data"), dataset_name)

    # test_text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    # At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the preparation and manipulation of quantum states.
    # Classical physics cannot explain the operation of these quantum devices, and a scalable quantum computer could perform some calculations exponentially faster (with respect to input size scaling)[2] than any modern "classical" computer. In particular, a large-scale quantum computer could break widely used encryption schemes and aid physicists in performing physical simulations; however, the current state of the technology is largely experimental and impractical, with several obstacles to useful applications. Moreover, scalable quantum computers do not hold promise for many practical tasks, and for many important tasks quantum speedups are proven impossible.
    # The basic unit of information in quantum computing is the qubit, similar to the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis" states. When measuring a qubit, the result is a probabilistic output of a classical bit, therefore making quantum computers nondeterministic in general. If a quantum computer manipulates the qubit in a particular way, wave interference effects can amplify the desired measurement results. The design of quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently and quickly.
    # Physically engineering high-quality qubits has proven challenging. If a physical qubit is not sufficiently isolated from its environment, it suffers from quantum decoherence, introducing noise into calculations. Paradoxically, perfectly isolating qubits is also undesirable because quantum computations typically need to initialize qubits, perform controlled qubit interactions, and measure the resulting quantum states. Each of those operations introduces errors and suffers from noise, and such inaccuracies accumulate.
    # In principle, a non-quantum (classical) computer can solve the same computational problems as a quantum computer, given enough time. Quantum advantage comes in the form of time complexity rather than computability, and quantum complexity theory shows that some quantum algorithms for carefully selected tasks require exponentially fewer computational steps than the best known non-quantum algorithms. Such tasks can in theory be solved on a large-scale quantum computer whereas classical computers would not finish computations in any reasonable amount of time. However, quantum speedup is not universal or even typical across computational tasks, since basic tasks such as sorting are proven to not allow any asymptotic quantum speedup. Claims of quantum supremacy have drawn significant attention to the discipline, but are demonstrated on contrived tasks, while near-term practical use cases remain limited.
    # """

    # dataset_name = "pravilnik.energetska efikasnost"
    # await add(test_text, dataset_name)

    graph = await cognify(dataset_name)

    await render_graph(graph, graph_type = "networkx")




if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



import os
import cognee
import pathlib

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.modules.engine.operations.setup import setup
from cognee.shared.logging_utils import setup_logging, CRITICAL

logger = get_logger()


async def main():
    # ENABLE PERMISSIONS FEATURE
    # Note: When ENABLE_BACKEND_ACCESS_CONTROL is enabled vector provider is automatically set to use LanceDB
    # and graph provider is set to use Kuzu.
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # Set up the necessary databases and tables for user management.
    await setup()

    # NOTE: When a document is added in Cognee with permissions enabled only the owner of the document has permissions
    # for work with the document initially.
    # Add document for user_1, add it under dataset name AI
    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "../data/artificial_intelligence.pdf"
    )

    print("Creating user_1: user_1@example.com")
    user_1 = await create_user("user_1@example.com", "example")
    await cognee.add([explanation_file_path], dataset_name="AI", user=user_1)

    # Add document for user_2, add it under dataset name QUANTUM
    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the preparation and manipulation of quantum states.
    Classical physics cannot explain the operation of these quantum devices, and a scalable quantum computer could perform some calculations exponentially faster (with respect to input size scaling) than any modern "classical" computer. In particular, a large-scale quantum computer could break widely used encryption schemes and aid physicists in performing physical simulations; however, the current state of the technology is largely experimental and impractical, with several obstacles to useful applications. Moreover, scalable quantum computers do not hold promise for many practical tasks, and for many important tasks quantum speedups are proven impossible.
    The basic unit of information in quantum computing is the qubit, similar to the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis" states. When measuring a qubit, the result is a probabilistic output of a classical bit, therefore making quantum computers nondeterministic in general. If a quantum computer manipulates the qubit in a particular way, wave interference effects can amplify the desired measurement results. The design of quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently and quickly.
    Physically engineering high-quality qubits has proven challenging. If a physical qubit is not sufficiently isolated from its environment, it suffers from quantum decoherence, introducing noise into calculations. Paradoxically, perfectly isolating qubits is also undesirable because quantum computations typically need to initialize qubits, perform controlled qubit interactions, and measure the resulting quantum states. Each of those operations introduces errors and suffers from noise, and such inaccuracies accumulate.
    In principle, a non-quantum (classical) computer can solve the same computational problems as a quantum computer, given enough time. Quantum advantage comes in the form of time complexity rather than computability, and quantum complexity theory shows that some quantum algorithms for carefully selected tasks require exponentially fewer computational steps than the best known non-quantum algorithms. Such tasks can in theory be solved on a large-scale quantum computer whereas classical computers would not finish computations in any reasonable amount of time. However, quantum speedup is not universal or even typical across computational tasks, since basic tasks such as sorting are proven to not allow any asymptotic quantum speedup. Claims of quantum supremacy have drawn significant attention to the discipline, but are demonstrated on contrived tasks, while near-term practical use cases remain limited.
    """
    print("\nCreating user_2: user_2@example.com")
    user_2 = await create_user("user_2@example.com", "example")
    await cognee.add([text], dataset_name="QUANTUM", user=user_2)

    # Run cognify for both datasets as the appropriate user/owner
    print("\nCreating different datasets for user_1 (AI dataset) and user_2 (QUANTUM dataset)")
    ai_cognify_result = await cognee.cognify(["AI"], user=user_1)
    quantum_cognify_result = await cognee.cognify(["QUANTUM"], user=user_2)

    # Extract dataset_ids from cognify results
    def extract_dataset_id_from_cognify(cognify_result):
        """Extract dataset_id from cognify output dictionary"""
        for dataset_id, pipeline_result in cognify_result.items():
            return dataset_id  # Return the first dataset_id
        return None

    # Get dataset IDs from cognify results
    # Note: When we want to use datasets from other users we must supply dataset information through dataset_id
    # Using dataset name only looks for datasets owned by current user
    ai_dataset_id = extract_dataset_id_from_cognify(ai_cognify_result)
    quantum_dataset_id = extract_dataset_id_from_cognify(quantum_cognify_result)

    # We can see here that user_1 can read his own dataset (AI dataset)
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=user_1,
        datasets=[ai_dataset_id],
    )
    print("\nSearch results as user_1 on dataset owned by user_1:")
    for result in search_results:
        print(f"{result}\n")

    # But user_1 cant read the dataset owned by user_2 (QUANTUM dataset)
    print("\nSearch result as user_1 on the dataset owned by user_2:")
    try:
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text="What is in the document?",
            user=user_1,
            datasets=[quantum_dataset_id],
        )
    except PermissionDeniedError:
        print(f"User: {user_1} does not have permission to read from dataset: QUANTUM")

    # user_1 currently also cant add a document to user_2's dataset (QUANTUM dataset)
    print("\nAttempting to add new data as user_1 to dataset owned by user_2:")
    try:
        await cognee.add(
            [explanation_file_path],
            dataset_id=quantum_dataset_id,
            user=user_1,
        )
    except PermissionDeniedError:
        print(f"User: {user_1} does not have permission to write to dataset: QUANTUM")

    # We've shown that user_1 can't interact with the dataset from user_2
    # Now have user_2 give proper permission to user_1 to read QUANTUM dataset
    # Note: supported permission types are "read", "write", "delete" and "share"
    print(
        "\nOperation started as user_2 to give read permission to user_1 for the dataset owned by user_2"
    )
    await authorized_give_permission_on_datasets(
        user_1.id,
        [quantum_dataset_id],
        "read",
        user_2.id,
    )

    # Now user_1 can read from quantum dataset after proper permissions have been assigned by the Quantum dataset owner.
    print("\nSearch result as user_1 on the dataset owned by user_2:")
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=user_1,
        dataset_ids=[quantum_dataset_id],
    )
    for result in search_results:
        print(f"{result}\n")

    # If we'd like for user_1 to add new documents to the quantum dataset from user_2 he'd have to get "write" access permission,
    # which he currently does not have


if __name__ == "__main__":
    import asyncio

    logger = setup_logging(log_level=CRITICAL)
    asyncio.run(main())

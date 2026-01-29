import os
import cognee
import pathlib
import asyncio

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.modules.engine.operations.setup import setup
from cognee.shared.logging_utils import setup_logging, CRITICAL

# ENABLE PERMISSIONS FEATURE
# Note: When ENABLE_BACKEND_ACCESS_CONTROL is enabled vector provider is automatically set to use LanceDB
# and graph provider is set to use Kuzu.
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

logger = get_logger()

explanation_file_path = os.path.join(
    pathlib.Path(__file__).parent, "data/artificial_intelligence.pdf"
)

text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages
this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the
preparation and manipulation of quantum states.
"""


# Extract dataset_ids from cognify results
def extract_dataset_id_from_cognify(cognify_result):
    """Extract dataset_id from cognify output dictionary"""
    return next(iter(cognify_result), None)  # Return the first dataset_id


async def data_access_control_example():
    # NOTE: When a document is added in Cognee with permissions enabled only the owner of the document has permissions
    # to work with the document initially.
    print("Creating user_1: user_1@example.com")
    user_1 = await create_user("user_1@example.com", "example")
    await cognee.add([explanation_file_path], dataset_name="AI", user=user_1)

    print("\nCreating user_2: user_2@example.com")
    user_2 = await create_user("user_2@example.com", "example")
    await cognee.add([text], dataset_name="QUANTUM", user=user_2)

    print("\nCreating different datasets for user_1 (AI dataset) and user_2 (QUANTUM dataset)")
    ai_cognify_result = await cognee.cognify(["AI"], user=user_1)
    quantum_cognify_result = await cognee.cognify(["QUANTUM"], user=user_2)

    # Get dataset IDs from cognify results
    # Note: When we want to work with datasets from other users (search, add, cognify and etc.) we must supply dataset
    # information through dataset_id using dataset name only looks for datasets owned by current user
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

    # user_1 currently also can't add a document to user_2's dataset (QUANTUM dataset)
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

    # Now user_1 can read from quantum dataset after proper permissions have been assigned by the QUANTUM dataset owner.
    print("\nSearch result as user_1 on the dataset owned by user_2:")
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=user_1,
        dataset_ids=[quantum_dataset_id],
    )
    for result in search_results:
        print(f"{result}\n")

    # If we'd like for user_1 to add new documents to the QUANTUM dataset owned by user_2, user_1 would have to get
    # "write" access permission, which user_1 currently does not have


async def main():
    # Create a clean slate for cognee -- reset data and system state and
    # set up the necessary databases and tables for user management.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    await data_access_control_example()


if __name__ == "__main__":
    logger = setup_logging(log_level=CRITICAL)
    asyncio.run(main())

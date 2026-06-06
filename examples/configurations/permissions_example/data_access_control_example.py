import asyncio
import os
import pathlib
from uuid import UUID

# ENABLE PERMISSIONS FEATURE
# Note: When ENABLE_BACKEND_ACCESS_CONTROL is enabled, vector provider is automatically set to use LanceDB.
# The default graph provider is Ladybug (can be overridden via GRAPH_DATABASE_PROVIDER env var).
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

import cognee
from cognee import SearchType
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.shared.logging_utils import CRITICAL, get_logger, setup_logging

logger = get_logger()

explanation_file_path = os.path.join(
    pathlib.Path(__file__).parent, "data/artificial_intelligence.pdf"
)

text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages
this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the
preparation and manipulation of quantum states.
"""


def get_dataset_id(remember_result):
    """Extract dataset_id from remember output."""
    return UUID(remember_result.dataset_id)


async def data_access_control_example():
    # NOTE: When a document is remembered in Cognee with permissions enabled only the owner of the document has permissions
    # to work with the document initially.
    print("Creating user_1: user_1@example.com")
    user_1 = await create_user("user_1@example.com", "example")
    ai_remember_result = await cognee.remember(
        [explanation_file_path],
        dataset_name="AI",
        user=user_1,
        self_improvement=False,
    )

    print("\nCreating user_2: user_2@example.com")
    user_2 = await create_user("user_2@example.com", "example")
    quantum_remember_result = await cognee.remember(
        [text],
        dataset_name="QUANTUM",
        user=user_2,
        self_improvement=False,
    )

    # Get dataset IDs from remember results
    # Note: When we want to work with datasets from other users (recall, remember, and etc.) we must supply dataset
    # information through dataset_ids; using dataset names only looks for datasets owned by current user
    ai_dataset_id = get_dataset_id(ai_remember_result)
    quantum_dataset_id = get_dataset_id(quantum_remember_result)

    # We can see here that user_1 can read his own dataset (AI dataset)
    recall_results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=user_1,
        dataset_ids=[ai_dataset_id],
    )
    print("\nRecall results as user_1 on dataset owned by user_1:")
    for result in recall_results:
        print(f"{result}\n")

    # But user_1 cant read the dataset owned by user_2 (QUANTUM dataset)
    print("\nRecall result as user_1 on the dataset owned by user_2:")
    try:
        await cognee.recall(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text="What is in the document?",
            user=user_1,
            dataset_ids=[quantum_dataset_id],
        )
    except PermissionDeniedError:
        print(f"User: {user_1} does not have permission to read from dataset: QUANTUM")

    # user_1 currently also can't remember a document to user_2's dataset (QUANTUM dataset)
    print("\nAttempting to remember new data as user_1 to dataset owned by user_2:")
    try:
        await cognee.remember(
            [explanation_file_path],
            dataset_id=quantum_dataset_id,
            user=user_1,
            self_improvement=False,
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
    print("\nRecall result as user_1 on the dataset owned by user_2:")
    recall_results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=user_1,
        dataset_ids=[quantum_dataset_id],
    )
    for result in recall_results:
        print(f"{result}\n")

    # If we'd like for user_1 to remember new documents to the QUANTUM dataset owned by user_2, user_1 would have to get
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

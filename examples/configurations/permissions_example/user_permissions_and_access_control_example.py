# ruff: noqa: E402
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
from cognee.modules.users.roles.methods import add_user_to_role, create_role
from cognee.modules.users.tenants.methods import add_user_to_tenant, create_tenant, select_tenant
from cognee.shared.logging_utils import CRITICAL, get_logger, setup_logging

logger = get_logger()


async def main():
    # Set the rest of your environment variables as needed. By default OpenAI is used as the LLM provider
    # Reference the .env.tempalte file for available option and how to change LLM provider: https://github.com/topoteretes/cognee/blob/main/.env.template
    # For example to set your OpenAI LLM API key use:
    # os.environ["LLM_API_KEY"] = "your-api-key"

    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # Set up the necessary databases and tables for user management.
    await setup()

    # NOTE: When a document is remembered in Cognee with permissions enabled only the owner of the document has permissions
    # to work with the document initially.
    # Remember document for user_1 under dataset name AI
    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "data/artificial_intelligence.pdf"
    )

    print("Creating user_1: user_1@example.com")
    user_1 = await create_user("user_1@example.com", "example")
    ai_remember_result = await cognee.remember(
        [explanation_file_path],
        dataset_name="AI",
        user=user_1,
        self_improvement=False,
    )

    # Remember document for user_2 under dataset name QUANTUM
    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages
    this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the
    preparation and manipulation of quantum states.
    """
    print("\nCreating user_2: user_2@example.com")
    user_2 = await create_user("user_2@example.com", "example")
    quantum_remember_result = await cognee.remember(
        [text],
        dataset_name="QUANTUM",
        user=user_2,
        self_improvement=False,
    )

    def get_dataset_id(remember_result):
        """Extract dataset_id from remember output."""
        return UUID(remember_result.dataset_id)

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

    # Users can also be added to Roles and Tenants and then permission can be assigned on a Role/Tenant level as well
    # To create a Role a user first must be an owner of a Tenant
    print("User 2 is creating CogneeLab tenant/organization")
    tenant_id = await create_tenant("CogneeLab", user_2.id)

    print("User 2 is selecting CogneeLab tenant/organization as active tenant")
    await select_tenant(user_id=user_2.id, tenant_id=tenant_id)

    print("\nUser 2 is creating Researcher role")
    role_id = await create_role(role_name="Researcher", owner_id=user_2.id)

    print("\nCreating user_3: user_3@example.com")
    user_3 = await create_user("user_3@example.com", "example")

    # To add a user to a role he must be part of the same tenant/organization
    print("\nOperation started as user_2 to add user_3 to CogneeLab tenant/organization")
    await add_user_to_tenant(user_id=user_3.id, tenant_id=tenant_id, owner_id=user_2.id)

    print(
        "\nOperation started by user_2, as tenant owner, to add user_3 to Researcher role inside the tenant/organization"
    )
    await add_user_to_role(user_id=user_3.id, role_id=role_id, owner_id=user_2.id)

    print("\nOperation as user_3 to select CogneeLab tenant/organization as active tenant")
    await select_tenant(user_id=user_3.id, tenant_id=tenant_id)

    print(
        "\nOperation started as user_2, with CogneeLab as its active tenant, to give read permission to Researcher role for the dataset QUANTUM owned by user_2"
    )
    # Even though the dataset owner is user_2, the dataset doesn't belong to the tenant/organization CogneeLab.
    # So we can't assign permissions to it when we're acting in the CogneeLab tenant.
    try:
        await authorized_give_permission_on_datasets(
            role_id,
            [quantum_dataset_id],
            "read",
            user_2.id,
        )
    except PermissionDeniedError:
        print(
            "User 2 could not give permission to the role as the QUANTUM dataset is not part of the CogneeLab tenant"
        )

    print(
        "We will now create a new QUANTUM dataset with the QUANTUM_COGNEE_LAB name in the CogneeLab tenant so that permissions can be assigned to the Researcher role inside the tenant/organization"
    )
    # We can re-create the QUANTUM dataset in the CogneeLab tenant. The old QUANTUM dataset is still owned by user_2 personally
    # and can still be accessed by selecting the personal tenant for user 2.
    from cognee.modules.users.methods import get_user

    # Note: We need to update user_2 from the database to refresh its tenant context changes
    user_2 = await get_user(user_2.id)
    quantum_cognee_lab_remember_result = await cognee.remember(
        [text],
        dataset_name="QUANTUM_COGNEE_LAB",
        user=user_2,
        self_improvement=False,
    )

    # The recreated Quantum dataset will now have a different dataset_id as it's a new dataset in a different organization
    quantum_cognee_lab_dataset_id = get_dataset_id(quantum_cognee_lab_remember_result)
    print(
        "\nOperation started as user_2, with CogneeLab as its active tenant, to give read permission to Researcher role for the dataset QUANTUM owned by the CogneeLab tenant"
    )
    await authorized_give_permission_on_datasets(
        role_id,
        [quantum_cognee_lab_dataset_id],
        "read",
        user_2.id,
    )

    # Now user_3 can read from QUANTUM dataset as part of the Researcher role after proper permissions have been assigned by the QUANTUM dataset owner, user_2.
    print("\nRecall result as user_3 on the QUANTUM dataset owned by the CogneeLab organization:")
    recall_results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=user_3,
        dataset_ids=[quantum_cognee_lab_dataset_id],
    )
    for result in recall_results:
        print(f"{result}\n")

    # Note: All of these function calls and permission system is available through our backend endpoints as well


if __name__ == "__main__":
    import asyncio

    logger = setup_logging(log_level=CRITICAL)
    asyncio.run(main())

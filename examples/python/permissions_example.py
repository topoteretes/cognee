import os
import cognee
import pathlib
from pprint import pprint

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.tenants.methods import select_tenant
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.modules.users.roles.methods import add_user_to_role
from cognee.modules.users.roles.methods import create_role
from cognee.modules.users.tenants.methods import create_tenant
from cognee.modules.users.tenants.methods import add_user_to_tenant
from cognee.modules.engine.operations.setup import setup
from cognee.shared.logging_utils import setup_logging, CRITICAL

logger = get_logger()


async def main():
    # ENABLE PERMISSIONS FEATURE
    # Note: When ENABLE_BACKEND_ACCESS_CONTROL is enabled vector provider is automatically set to use LanceDB
    # and graph provider is set to use Kuzu.
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    # Set the rest of your environment variables as needed. By default OpenAI is used as the LLM provider
    # Reference the .env.tempalte file for available option and how to change LLM provider: https://github.com/topoteretes/cognee/blob/main/.env.template
    # For example to set your OpenAI LLM API key use:
    # os.environ["LLM_API_KEY""] = "your-api-key"

    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # Set up the necessary databases and tables for user management.
    await setup()

    # NOTE: When a document is added in Cognee with permissions enabled only the owner of the document has permissions
    # to work with the document initially.
    # Add document for user_1, add it under dataset name AI
    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "../data/artificial_intelligence.pdf"
    )

    print("Creating user_1: user_1@example.com")
    user_1 = await create_user("user_1@example.com", "example")
    await cognee.add([explanation_file_path], dataset_name="AI", user=user_1)

    # Add document for user_2, add it under dataset name QUANTUM
    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages
    this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the
    preparation and manipulation of quantum states.
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
        pprint(result)

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
        pprint(result)

    # If we'd like for user_1 to add new documents to the QUANTUM dataset owned by user_2, user_1 would have to get
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
    await cognee.add([text], dataset_name="QUANTUM_COGNEE_LAB", user=user_2)
    quantum_cognee_lab_cognify_result = await cognee.cognify(["QUANTUM_COGNEE_LAB"], user=user_2)

    # The recreated Quantum dataset will now have a different dataset_id as it's a new dataset in a different organization
    quantum_cognee_lab_dataset_id = extract_dataset_id_from_cognify(
        quantum_cognee_lab_cognify_result
    )
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
    print("\nSearch result as user_3 on the QUANTUM dataset owned by the CogneeLab organization:")
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=user_3,
        dataset_ids=[quantum_cognee_lab_dataset_id],
    )
    for result in search_results:
        pprint(result)

    # Note: All of these function calls and permission system is available through our backend endpoints as well


if __name__ == "__main__":
    import asyncio

    logger = setup_logging(log_level=CRITICAL)
    asyncio.run(main())

import os
import cognee

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
from cognee.modules.users.methods import get_user

# ENABLE PERMISSIONS FEATURE
# Note: When ENABLE_BACKEND_ACCESS_CONTROL is enabled vector provider is automatically set to use LanceDB
# and graph provider is set to use Kuzu.
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

logger = get_logger()

text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages
this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the
preparation and manipulation of quantum states.
"""


# Extract dataset_ids from cognify results
def extract_dataset_id_from_cognify(cognify_result):
    """Extract dataset_id from cognify output dictionary"""
    return next(iter(cognify_result), None)  # Return the first dataset_id


async def tenant_and_role_setup_example():
    # NOTE: When a document is added in Cognee with permissions enabled only the owner of the document has permissions
    # to work with the document initially.

    # Add document for user_1, add it under dataset name QUANTUM
    print("\nCreating user_1: user_1@example.com")
    user_1 = await create_user("user_1@example.com", "example")
    await cognee.add([text], dataset_name="QUANTUM", user=user_1)

    # Users can also be added to Roles and Tenants and then permission can be assigned on a Role/Tenant level as well
    # To create a Role a user first must be an owner of a Tenant
    print("User 1 is creating CogneeLab tenant/organization")
    tenant_id = await create_tenant("CogneeLab", user_1.id)

    print("User 1 is selecting CogneeLab tenant/organization as active tenant")
    await select_tenant(user_id=user_1.id, tenant_id=tenant_id)

    print("\nUser 1 is creating Researcher role")
    role_id = await create_role(role_name="Researcher", owner_id=user_1.id)

    print("\nCreating user_2: user_2@example.com")
    user_2 = await create_user("user_2@example.com", "example")

    # To add a user to a role he must be part of the same tenant/organization
    print("\nOperation started as user_1 to add user_2 to CogneeLab tenant/organization")
    await add_user_to_tenant(user_id=user_2.id, tenant_id=tenant_id, owner_id=user_1.id)

    print(
        "\nOperation started by user_1, as tenant owner, to add user_2 to Researcher role inside the tenant/organization"
    )
    await add_user_to_role(user_id=user_2.id, role_id=role_id, owner_id=user_1.id)

    print("\nOperation as user_2 to select CogneeLab tenant/organization as active tenant")
    await select_tenant(user_id=user_2.id, tenant_id=tenant_id)

    # Note: We need to update user_1 from the database to refresh its tenant context changes
    user_1 = await get_user(user_1.id)
    await cognee.add([text], dataset_name="QUANTUM_COGNEE_LAB", user=user_1)
    quantum_cognee_lab_cognify_result = await cognee.cognify(["QUANTUM_COGNEE_LAB"], user=user_1)

    quantum_cognee_lab_dataset_id = extract_dataset_id_from_cognify(
        quantum_cognee_lab_cognify_result
    )
    print(
        "\nOperation started as user_1, with CogneeLab as its active tenant, to give read permission to Researcher role for the dataset QUANTUM owned by the CogneeLab tenant"
    )
    await authorized_give_permission_on_datasets(
        role_id,
        [quantum_cognee_lab_dataset_id],
        "read",
        user_1.id,
    )

    # Now user_2 can read from QUANTUM dataset as part of the Researcher role after proper permissions have been assigned by the QUANTUM dataset owner, user_1.
    print("\nSearch result as user_2 on the QUANTUM dataset owned by the CogneeLab organization:")
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=user_2,
        dataset_ids=[quantum_cognee_lab_dataset_id],
    )
    for result in search_results:
        print(f"{result}\n")


async def main():
    # Create a clean slate for cognee -- reset data and system state and
    # set up the necessary databases and tables for user management.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    await tenant_and_role_setup_example()

    # Note: All of these function calls and permission system is available through our backend endpoints as well


if __name__ == "__main__":
    import asyncio

    logger = setup_logging(log_level=CRITICAL)
    asyncio.run(main())

import asyncio
from uuid import UUID

import cognee
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.modules.users.roles.methods import add_user_to_role, create_role
from cognee.modules.users.tenants.methods import add_user_to_tenant, create_tenant, select_tenant
from cognee.shared.logging_utils import CRITICAL, get_logger, setup_logging

logger = get_logger()

text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages
this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the
preparation and manipulation of quantum states.
"""


def get_dataset_id(remember_result):
    """Extract dataset_id from remember output."""
    return UUID(remember_result.dataset_id)


async def tenant_and_role_constraints_example():
    # NOTE: When a document is remembered in Cognee with permissions enabled only the owner of the document has permissions
    # to work with the document initially.

    # Remember document for user_1 under dataset name QUANTUM
    print("\nCreating user_1: user_1@example.com")
    user_1 = await create_user("user_1@example.com", "example")
    quantum_remember_result = await cognee.remember(
        [text],
        dataset_name="QUANTUM",
        user=user_1,
        self_improvement=False,
    )

    # Get dataset IDs from remember results
    # Note: When we want to work with datasets from other users (recall, remember, and etc.) we must supply dataset
    # information through dataset_ids; using dataset names only looks for datasets owned by current user
    quantum_dataset_id = get_dataset_id(quantum_remember_result)

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

    print(
        "\nOperation started as user_1, with CogneeLab as its active tenant, to give read permission to Researcher role for the dataset QUANTUM owned by user_1"
    )
    # Even though the dataset owner is user_1, the dataset doesn't belong to the tenant/organization CogneeLab.
    # So we can't assign permissions to it when we're acting in the CogneeLab tenant.
    try:
        await authorized_give_permission_on_datasets(
            role_id,
            [quantum_dataset_id],
            "read",
            user_1.id,
        )
    except PermissionDeniedError:
        print(
            "User 1 could not give permission to the role as the QUANTUM dataset is not part of the CogneeLab tenant"
        )

    # We can re-create the QUANTUM dataset in the CogneeLab tenant. The old QUANTUM dataset is still owned by user_1 personally


async def main():
    # Create a clean slate for cognee -- reset data and system state and
    # set up the necessary databases and tables for user management.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    await tenant_and_role_constraints_example()

    # Note: All of these function calls and permission system is available through our backend endpoints as well


# Please set ENABLE_BACKEND_ACCESS_CONTROL=True in .env file
# Note: When ENABLE_BACKEND_ACCESS_CONTROL is enabled, vector provider is automatically set to use LanceDB.
# The default graph provider is Ladybug (can be overridden via GRAPH_DATABASE_PROVIDER env var).
if __name__ == "__main__":
    logger = setup_logging(log_level=CRITICAL)
    asyncio.run(main())

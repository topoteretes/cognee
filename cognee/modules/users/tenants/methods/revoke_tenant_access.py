from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.methods.get_agent_user_ids import get_agent_user_ids
from cognee.modules.users.models import User
from cognee.modules.users.models.ACL import ACL
from cognee.modules.users.models.PrincipalCapability import PrincipalCapability
from cognee.modules.users.models.Role import Role
from cognee.modules.users.models.UserApiKey import UserApiKey
from cognee.modules.users.models.UserRole import UserRole
from cognee.modules.users.models.UserTenant import UserTenant


async def revoke_tenant_access(session: AsyncSession, user_id: UUID, tenant_id: UUID) -> None:
    """Take a user's access to a tenant away, and that of the agents they created.

    An agent is a separate user with its own keys, ACLs and memberships, so
    ending the parent's access never reached it. This covers the user and every
    agent descended from them: tenant membership, roles in the tenant, ACLs on
    the tenant's datasets and capabilities granted in the tenant are deleted,
    and the API keys of the agents working in the tenant. An agent working in
    another tenant keeps its keys, since it cannot reach this one. Those who
    had the tenant as their current tenant lose that too, otherwise a new key
    or a new dataset would put them back in it. It is a function of its own so
    that the Cloud pod's removal path can call it instead of repeating the walk
    over the agents. The caller owns the transaction.
    """
    agent_ids = await get_agent_user_ids(session, user_id)
    principal_ids = [user_id, *agent_ids]

    # Remove the user-tenant association first. add_user_to_role locks this
    # row while it inserts a role membership, so on Postgres a concurrent
    # assignment either waits for this removal and then finds no
    # membership, or finishes first and has its row removed below.
    await session.execute(
        delete(UserTenant).where(
            UserTenant.user_id.in_(principal_ids),
            UserTenant.tenant_id == tenant_id,
        )
    )

    # Subquery for role ids in this tenant
    role_ids_in_tenant = select(Role.id).where(Role.tenant_id == tenant_id)
    # Remove them from all roles in this tenant
    await session.execute(
        delete(UserRole).where(
            UserRole.user_id.in_(principal_ids),
            UserRole.role_id.in_(role_ids_in_tenant),
        )
    )

    # Subquery for dataset ids in this tenant
    dataset_ids_in_tenant = select(Dataset.id).where(Dataset.tenant_id == tenant_id)
    # Revoke their permissions on datasets in this tenant
    await session.execute(
        delete(ACL).where(
            ACL.principal_id.in_(principal_ids),
            ACL.dataset_id.in_(dataset_ids_in_tenant),
        )
    )

    # Revoke capabilities granted to them personally in this tenant.
    # Resolution already ignores them while the user is not a member, but
    # left in place they would come back if the user is added again, so a
    # requester with only MANAGE_USERS could remove and re-add someone to
    # restore capabilities they cannot grant themselves.
    await session.execute(
        delete(PrincipalCapability).where(
            PrincipalCapability.principal_id.in_(principal_ids),
            PrincipalCapability.tenant_id == tenant_id,
        )
    )

    # Before the current tenant is cleared below, which is what selects them.
    await session.execute(
        delete(UserApiKey).where(
            UserApiKey.user_id.in_(
                select(User.id).where(User.id.in_(agent_ids), User.tenant_id == tenant_id)
            )
        )
    )

    await session.execute(
        update(User)
        .where(User.id.in_(principal_ids), User.tenant_id == tenant_id)
        .values(tenant_id=None)
    )

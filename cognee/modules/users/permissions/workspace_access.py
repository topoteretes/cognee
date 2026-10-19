"""Read back persisted grants and distinguish them from inherited access."""

from uuid import UUID

from sqlalchemy import delete, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset
from cognee.modules.users.models import ACL, Permission, Role, Tenant, User, UserRole, UserTenant
from cognee.modules.users.permissions.methods.give_permission_on_dataset import (
    give_permission_on_dataset,
)

PERMISSIONS = {"read", "write", "share", "delete"}


async def dataset_access(user: User, dataset_id: UUID) -> dict:
    dataset = await get_authorized_dataset(user, dataset_id, "share")
    if dataset is None or dataset.tenant_id != user.tenant_id:
        raise PermissionError("You cannot manage access to this dataset")
    async with get_relational_engine().get_async_session() as session:
        tenant = await session.get(Tenant, dataset.tenant_id) if dataset.tenant_id else None
        # Only the team owner may enumerate the whole membership directory.
        # Other sharers may manage their own agents and existing grantees.
        grants = (
            await session.execute(
                select(ACL.principal_id, Permission.name)
                .join(Permission, Permission.id == ACL.permission_id)
                .where(ACL.dataset_id == dataset_id)
            )
        ).all()
        direct: dict[UUID, set[str]] = {}
        for principal_id, name in grants:
            direct.setdefault(principal_id, set()).add(name)
        team_members = set(
            (
                await session.execute(
                    select(UserTenant.user_id).where(UserTenant.tenant_id == dataset.tenant_id)
                )
            )
            .scalars()
            .all()
        )
        # Plugin creation sets the active tenant but does not enroll the agent
        # as a team member. Those agents still need explicit dataset grants.
        members = select(User).where(
            User.id.in_(team_members)
            | ((User.parent_user_id == user.id) & (User.tenant_id == dataset.tenant_id))
        )
        if tenant is None or tenant.owner_id != user.id:
            members = members.where(
                (User.id == user.id) | (User.parent_user_id == user.id) | User.id.in_(direct)
            )
        users = list((await session.execute(members)).scalars().all())
        if not any(member.id == user.id for member in users):
            users.append(user)
        roles = list(
            (await session.execute(select(Role).where(Role.tenant_id == dataset.tenant_id)))
            .scalars()
            .all()
        )
        memberships = (
            await session.execute(
                select(UserRole.user_id, UserRole.role_id).where(
                    UserRole.user_id.in_([member.id for member in users]),
                    UserRole.role_id.in_([role.id for role in roles]),
                )
            )
        ).all()
        rows = []
        for member in users:
            inherited = (
                set(direct.get(dataset.tenant_id, set())) if member.id in team_members else set()
            )
            for member_id, role_id in memberships:
                if member_id == member.id and member.id in team_members:
                    inherited.update(direct.get(role_id, set()))
            own = direct.get(member.id, set())
            rows.append(
                {
                    "id": member.id,
                    "name": member.email,
                    "kind": "agent" if member.parent_user_id else "person",
                    "direct": sorted(own),
                    "inherited": sorted(inherited),
                    "effective": sorted(own | inherited),
                    "owner": member.id == dataset.owner_id,
                }
            )
        for principal, kind in [(role, "role") for role in roles] + (
            [(tenant, "team")] if tenant else []
        ):
            own = sorted(direct.get(principal.id, set()))
            rows.append(
                {
                    "id": principal.id,
                    "name": principal.name,
                    "kind": kind,
                    "direct": own,
                    "inherited": [],
                    "effective": own,
                    "owner": False,
                }
            )
        return {"dataset_id": dataset.id, "principals": rows}


async def set_direct_access(
    user: User, dataset_id: UUID, principal_id: UUID, permission: str, allowed: bool
):
    if permission not in PERMISSIONS:
        raise ValueError("Unknown dataset permission")
    listing = await dataset_access(user, dataset_id)
    principal = next((row for row in listing["principals"] if row["id"] == principal_id), None)
    if principal is None:
        raise PermissionError("Principal is not available in this team")
    if principal["owner"]:
        raise ValueError("The dataset owner's permissions cannot be changed here")
    # Mutate only the selected permission. A stale tab must not overwrite a
    # concurrent change to any of this principal's other permissions.
    async with get_relational_engine().get_async_session() as session:
        if allowed:
            from cognee.modules.users.models.Principal import Principal

            target = await session.get(Principal, principal_id)
            await give_permission_on_dataset(target, dataset_id, permission)
        else:
            await session.execute(
                delete(ACL).where(
                    ACL.dataset_id == dataset_id,
                    ACL.principal_id == principal_id,
                    ACL.permission_id.in_(
                        select(Permission.id).where(Permission.name == permission)
                    ),
                )
            )
            await session.commit()

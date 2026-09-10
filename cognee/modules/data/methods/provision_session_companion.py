"""Provision a companion on the server that owns the authoritative ACL table."""

from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset
from cognee.modules.users.models import ACL


class CompanionConflict(ValueError):
    pass


async def provision_session_companion(primary_id: UUID, user) -> dict:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        async with session.begin():
            primary = (
                await session.scalars(
                    select(Dataset).where(Dataset.id == primary_id).with_for_update()
                )
            ).first()
            # Owners alone can create or attest a complete ACL snapshot. Readers
            # must never infer sharing rights from list-datasets visibility.
            if (
                primary is None
                or primary.owner_id != user.id
                or primary.tenant_id != user.tenant_id
            ):
                raise PermissionError("Only the dataset owner can provision a companion")
            if primary.name.endswith("-agent_sessions"):
                raise CompanionConflict("Cannot provision a companion of a companion")
            companion_id = uuid5(NAMESPACE_URL, "cognee:session-companion:" + str(primary.id))
            name = primary.name + "-agent_sessions"
            companion = await session.get(Dataset, companion_id)
            rows = (await session.scalars(select(ACL).where(ACL.dataset_id == primary.id))).all()
            permissions = {(row.principal_id, row.permission_id) for row in rows}
            if not permissions:
                raise CompanionConflict("Primary dataset has no permission snapshot")
            if companion is None:
                collision = (
                    await session.scalars(
                        select(Dataset).where(
                            Dataset.name == name,
                            Dataset.owner_id == primary.owner_id,
                            Dataset.tenant_id == primary.tenant_id,
                        )
                    )
                ).first()
                if collision is not None:
                    raise CompanionConflict(
                        "Companion name is already used by an unrelated dataset"
                    )
                companion = Dataset(
                    id=companion_id,
                    name=name,
                    owner_id=primary.owner_id,
                    tenant_id=primary.tenant_id,
                )
                session.add(companion)
                await session.flush()
                for principal_id, permission_id in permissions:
                    session.add(
                        ACL(
                            dataset_id=companion_id,
                            principal_id=principal_id,
                            permission_id=permission_id,
                        )
                    )
            else:
                if (companion.name, companion.owner_id, companion.tenant_id) != (
                    name,
                    primary.owner_id,
                    primary.tenant_id,
                ):
                    raise CompanionConflict("Companion identity mismatch")
                existing = (
                    await session.scalars(select(ACL).where(ACL.dataset_id == companion_id))
                ).all()
                if {(row.principal_id, row.permission_id) for row in existing} != permissions:
                    # Do not silently mutate access to already captured conversations.
                    raise CompanionConflict("Companion permissions differ; reconcile access first")
        return {
            "primary_dataset_id": str(primary_id),
            "dataset_id": str(companion_id),
            "dataset_name": name,
            "permissions_verified": True,
        }

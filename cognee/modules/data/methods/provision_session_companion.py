"""Provision a dataset's ``<name>-agent_sessions`` companion with its ACL snapshot.

The agent plugins can route a session's Q&A and traces into a companion
dataset so the primary graph stays free of conversational chatter, and recall
across both. The companion must be readable by exactly the people who can
read the primary, and the authoritative ACL table lives on the server, so the
server provisions it: one transaction creates the companion and copies every
(principal, permission) pair the primary carries. A later call verifies the
snapshot instead of mutating it; drift is a 409 the client must resolve by
falling back to the primary, never by widening access on its own.

The companion is an ordinary dataset owned by the primary's owner, with the
id ``POST /datasets`` would derive for that owner and name, so creating it
through either path yields the same dataset.
"""

from uuid import UUID

from fastapi import status
from sqlalchemy import select

from cognee.exceptions import CogneeApiError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
from cognee.modules.data.models import Dataset
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import ACL, User

COMPANION_SUFFIX = "-agent_sessions"


class CompanionConflictError(CogneeApiError):
    """The companion exists but does not match the primary, or cannot be derived."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            name="CompanionConflictError",
            status_code=status.HTTP_409_CONFLICT,
            log=False,
        )


def companion_dataset_name(primary_name: str) -> str:
    return primary_name + COMPANION_SUFFIX


def may_provision(primary: Dataset, user: User) -> bool:
    """The owner, or an agent user whose parent is the owner, in the same tenant.

    Readers must not infer sharing rights from being able to list the dataset;
    only the owner (or an identity provisioned by the owner) attests the snapshot.
    """
    if primary.tenant_id != user.tenant_id:
        return False
    if primary.owner_id == user.id:
        return True
    return getattr(user, "parent_user_id", None) == primary.owner_id


async def _acl_snapshot(session, dataset_id: UUID) -> set[tuple[UUID, UUID]]:
    rows = (await session.scalars(select(ACL).where(ACL.dataset_id == dataset_id))).all()
    return {(row.principal_id, row.permission_id) for row in rows}


async def provision_session_companion(primary_id: UUID, user: User) -> dict:
    """Create or verify the companion of ``primary_id`` and return its identity.

    Raises ``PermissionDeniedError`` (403) when ``user`` may not provision it and
    ``CompanionConflictError`` (409) when a companion exists with different
    permissions or identity, when the primary is itself a companion, or when
    the primary has no permissions to copy.
    """
    engine = get_relational_engine()

    async with engine.get_async_session() as session:
        primary = await session.get(Dataset, primary_id)
        if primary is None or not may_provision(primary, user):
            raise PermissionDeniedError(
                "Only the dataset owner, or an agent of the owner, can provision a session companion"
            )
        if primary.name.endswith(COMPANION_SUFFIX):
            raise CompanionConflictError("Cannot provision a companion of a companion")
        owner = await session.get(User, primary.owner_id)
        if owner is None:
            raise CompanionConflictError("The primary dataset's owner no longer exists")
        name = companion_dataset_name(primary.name)

    # The id POST /datasets derives for this owner and name (read-only lookup,
    # done outside the write transaction below).
    companion_id = await get_unique_dataset_id(name, owner)

    async with engine.get_async_session() as session, session.begin():
        snapshot = await _acl_snapshot(session, primary_id)
        if not snapshot:
            raise CompanionConflictError("Primary dataset has no permissions to copy")

        companion = await session.get(Dataset, companion_id)
        if companion is None:
            companion = Dataset(
                id=companion_id,
                name=name,
                owner_id=primary.owner_id,
                tenant_id=primary.tenant_id,
            )
            session.add(companion)
            await session.flush()
            for principal_id, permission_id in snapshot:
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
                raise CompanionConflictError("Companion identity mismatch")
            if await _acl_snapshot(session, companion_id) != snapshot:
                # Never silently change who can read already captured sessions.
                raise CompanionConflictError(
                    "Companion permissions differ from the primary's; reconcile access first"
                )

    return {
        "primary_dataset_id": str(primary_id),
        "dataset_id": str(companion_id),
        "dataset_name": name,
        "permissions_verified": True,
    }

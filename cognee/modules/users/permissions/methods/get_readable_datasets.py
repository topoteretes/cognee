from uuid import UUID

from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)


async def get_readable_datasets(user_id: UUID) -> list[Dataset]:
    """Return the datasets ``user_id`` has read permission for (empty on none).

    ``get_specific_user_permission_datasets`` raises ``PermissionDeniedError``
    when a user has zero datasets with the requested permission — an
    ordinary, common state (most callers own no shared-with-them
    datasets), not a real authorization failure — so it's translated to
    an empty list here rather than propagated. Any other exception is
    left to propagate: a caller with a genuine problem (e.g. the DB is
    unreachable) should see that failure, not a silently empty result.
    """
    try:
        return await get_specific_user_permission_datasets(user_id, "read", None)
    except PermissionDeniedError:
        return []

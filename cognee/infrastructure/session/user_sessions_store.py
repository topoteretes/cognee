"""Storage home for user-scoped (custom-named) session vectors.

User sessions are cross-dataset by design, but in access-control mode vector
databases only exist per (dataset, owner) — there is no null context
(``apply_database_context_variables`` raises on ``dataset=None``). So every
user gets one reserved dataset, ``user_sessions``, used purely as the storage
home for their session vectors: one deterministic place to write, read, and
delete them regardless of which dataset context is active. The sessions
themselves stay unbound (``SessionRecord.dataset_id`` is NULL) — the reserved
dataset is a storage detail, not an attribution.

In single-user mode all stores are shared already, so this is a no-op.
"""

from contextlib import asynccontextmanager
from uuid import UUID

from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)

USER_SESSIONS_DATASET_NAME = "user_sessions"


@asynccontextmanager
async def user_sessions_store_context(user_id: str | UUID):
    """Enter the caller's reserved sessions-store context (no-op in single-user mode)."""
    if not backend_access_control_enabled():
        yield
        return

    from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
        resolve_authorized_user_datasets,
    )
    from cognee.modules.users.methods import get_user

    user = await get_user(UUID(str(user_id)))
    user, datasets = await resolve_authorized_user_datasets(USER_SESSIONS_DATASET_NAME, user)

    async with set_database_global_context_variables(datasets[0].id, user.id):
        yield

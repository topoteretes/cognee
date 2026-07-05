from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from uuid import UUID

from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets

from ...models.User import User

logger = get_logger()


import asyncio
from cognee.modules.governance.audit_repository import insert_audit_event

async def check_permission_on_dataset(user: User, permission_type: str, dataset_id: UUID):
    """
        Check if a user has a specific permission on a dataset.
    Args:
        user: User whose permission is checked
        permission_type: Type of permission to check
        dataset_id: Id of the dataset

    Returns:
        None

    """
    if user is None:
        user = await get_default_user()

    try:
        await get_specific_user_permission_datasets(user.id, permission_type, [dataset_id])
        
        asyncio.create_task(
            insert_audit_event(
                actor_id=user.id,
                action=permission_type,
                target_dataset_id=dataset_id,
                outcome="ALLOWED",
                policy_id=None,
                denial_reason=None,
            )
        )
    except Exception as e:
        asyncio.create_task(
            insert_audit_event(
                actor_id=user.id,
                action=permission_type,
                target_dataset_id=dataset_id,
                outcome="DENIED",
                policy_id=None,
                denial_reason=f"no ACL entry for action={permission_type} on dataset={dataset_id}",
            )
        )
        raise e

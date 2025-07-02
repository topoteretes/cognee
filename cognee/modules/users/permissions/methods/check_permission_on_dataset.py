from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from uuid import UUID

from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets

from ...models.User import User

logger = get_logger()


async def check_permission_on_dataset(user: User, permission_type: str, dataset_id: UUID):
    if user is None:
        user = await get_default_user()

    await get_specific_user_permission_datasets(user.id, permission_type, [dataset_id])

from cognee.shared.logging_utils import get_logger
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from cognee.infrastructure.databases.relational import get_relational_engine

from ...models.User import User
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.permissions.methods import get_principal_datasets

logger = get_logger()


async def get_all_user_permission_datasets(user: User, permission_type: str) -> list[Dataset]:
    datasets = list()
    # Get all datasets User has explicit access to
    datasets.extend(await get_principal_datasets(user, permission_type))
    # Get all datasets Users roles have access to
    # TODO: Expand to get all user role accessible datasets
    for role in user.roles:
        datasets.extend(await get_principal_datasets(role, permission_type))
    # Get all datasets Users tenant allows access for
    # TODO: Expand to get all user tenant accessible datasets
    if user.tenant_id:
        datasets.extend(await get_principal_datasets(user.tenant, permission_type))
    # TODO: Make sure result does not contain duplicate datasets
    return datasets

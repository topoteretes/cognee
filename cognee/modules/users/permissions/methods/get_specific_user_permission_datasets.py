from ...models.User import User
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets


async def get_specific_user_permission_datasets(
    user: User, permission_type: str, datasets: list[str] = None
) -> list[Dataset]:
    # Find all datasets user has permission for
    user_read_access_datasets = await get_all_user_permission_datasets(user, permission_type)

    # if specific datasets are provided filter out non provided datasets
    if datasets:
        search_datasets = [
            dataset for dataset in user_read_access_datasets if dataset.name in datasets
        ]
    else:
        search_datasets = user_read_access_datasets

    return search_datasets

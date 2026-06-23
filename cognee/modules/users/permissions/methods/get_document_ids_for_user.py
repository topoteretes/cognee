from uuid import UUID

from sqlalchemy import select

from cognee.modules.data.methods import get_dataset_data
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset
from ...models import ACL, Permission


async def get_document_ids_for_user(user_id: UUID, dataset_ids: list[UUID] = None) -> list[str]:
    """
        Return a list of document ids for which the user has read permission.
        If dataset_ids are specified, return only documents from those datasets
        (still restricted to the ones the user can actually read).
    Args:
        user_id: Id of the user
        dataset_ids: Optional list of dataset ids to restrict the result to

    Returns:
        list[str]: List of document ids for which the user has read permission
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        async with session.begin():
            # Datasets the user has read permission for, regardless of ownership.
            readable_dataset_ids = (
                await session.scalars(
                    select(Dataset.id)
                    .join(ACL.dataset)
                    .join(ACL.permission)
                    .where(
                        ACL.principal_id == user_id,
                        Permission.name == "read",
                    )
                )
            ).all()

            if dataset_ids is not None:
                # Keep only the requested datasets the user is allowed to read.
                requested = set(dataset_ids)
                readable_dataset_ids = [
                    dataset_id for dataset_id in readable_dataset_ids if dataset_id in requested
                ]

            document_ids = []
            for dataset_id in readable_dataset_ids:
                data_list = await get_dataset_data(dataset_id)
                document_ids.extend([data.id for data in data_list])

            return document_ids

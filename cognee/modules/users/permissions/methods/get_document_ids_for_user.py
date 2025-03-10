from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset, DatasetData, Data
from ...models import ACL, Permission


async def get_document_ids_for_user(user_id: UUID, datasets: list[str] = None) -> list[str]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        async with session.begin():
            document_ids = (
                await session.scalars(
                    select(Data.id)
                    .join(ACL.data)
                    .join(ACL.permission)
                    .where(
                        ACL.principal_id == user_id,
                        Permission.name == "read",
                    )
                )
            ).all()

            if datasets:
                documents_ids_in_dataset = set()
                # If datasets are specified filter out documents that aren't part of the specified datasets
                for dataset in datasets:
                    # Find dataset id for dataset element
                    dataset_id = (
                        await session.scalars(
                            select(Dataset.id).where(
                                Dataset.name == dataset,
                                Dataset.owner_id == user_id,
                            )
                        )
                    ).one_or_none()

                    # Check which documents are connected to this dataset
                    for document_id in document_ids:
                        data_id = (
                            await session.scalars(
                                select(DatasetData.data_id).where(
                                    DatasetData.dataset_id == dataset_id,
                                    DatasetData.data_id == document_id,
                                )
                            )
                        ).one_or_none()

                        # If document is related to dataset added it to return value
                        if data_id:
                            documents_ids_in_dataset.add(document_id)
                return list(documents_ids_in_dataset)
            return document_ids

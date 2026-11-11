from uuid import UUID

from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods.get_all_user_permission_datasets import (
    get_all_user_permission_datasets,
)


async def _get_existing_tenant_datasets(
    dataset_ids: list[UUID], tenant_id: UUID | None
) -> list[Dataset]:
    """Return existing Dataset rows from ``dataset_ids`` belonging to ``tenant_id``.

    Best-effort diagnostic lookup: returns an empty list if the relational
    engine is unavailable or mocked in unit tests.
    """
    if not dataset_ids:
        return []
    try:
        from sqlalchemy import select

        from cognee.infrastructure.databases.relational import get_relational_engine

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            rows = (
                await session.scalars(select(Dataset).where(Dataset.id.in_(dataset_ids)))
            ).all()
        return [row for row in rows if getattr(row, "tenant_id", None) == tenant_id]
    except Exception:
        return []


async def get_specific_user_permission_datasets(
    user_id: UUID, permission_type: str, dataset_ids: list[UUID] | None = None
) -> list[Dataset]:
    """
        Return a list of datasets user has given permission for. If a list of datasets is provided,
        verify for which datasets user has appropriate permission for and return list of datasets he has permission for.
    Args:
        user_id: Id of the user.
        permission_type: Type of the permission.
        dataset_ids: Ids of the provided datasets

    Returns:
        list[Dataset]: List of datasets user has permission for
    """
    user = await get_user(user_id)
    # Find all datasets user has permission for
    user_permission_access_datasets = await get_all_user_permission_datasets(user, permission_type)

    # if specific datasets are provided filter out non provided datasets
    if dataset_ids:
        search_datasets = [
            dataset for dataset in user_permission_access_datasets if dataset.id in dataset_ids
        ]
        # If there are requested datasets that user does not have access to raise error
        if len(search_datasets) != len(dataset_ids):
            permitted_ids = {dataset.id for dataset in search_datasets}
            denied_ids = [
                dataset_id
                for dataset_id in dict.fromkeys(dataset_ids)
                if dataset_id not in permitted_ids
            ]
            existing_denied = await _get_existing_tenant_datasets(
                denied_ids, getattr(user, "tenant_id", None)
            )
            base_message = (
                f"Request owner does not have necessary permission: [{permission_type}] "
                "for all datasets requested."
            )
            if existing_denied:
                details = ", ".join(
                    f"{ds.id} (name={ds.name!r}, created_at="
                    f"{ds.created_at.isoformat() if getattr(ds, 'created_at', None) else 'unknown'})"
                    for ds in existing_denied
                )
                raise PermissionDeniedError(
                    f"{base_message} Principal {user.id} has no [{permission_type}] ACL on "
                    f"dataset(s): {details} (if the dataset was deleted and recreated under "
                    "the same deterministic ID, non-owner ACLs were cleared and must be "
                    "granted again)."
                )
            raise PermissionDeniedError(base_message)
    else:
        search_datasets = user_permission_access_datasets

    if len(search_datasets) == 0:
        # A user with zero datasets is a normal state (fresh install, heartbeat
        # recall before first ingestion) and get_readable_datasets converts this
        # raise into an empty list — log at DEBUG so it doesn't surface as a
        # spurious ERROR on every such call. The line-36 raise above stays at
        # ERROR: requesting specific unpermitted datasets is a real denial.
        raise PermissionDeniedError(
            f"Request owner does not have permission: [{permission_type}] for any dataset.",
            log_level="DEBUG",
        )

    return search_datasets

from typing import Optional
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..exceptions import AmbiguousDataIdError, UnauthorizedDataAccessError
from ..models import Data, Dataset
from .resolve_data_id import resolve_data_id


async def get_data(
    user_id: UUID,
    data_id: UUID,
    dataset_id: Optional[UUID] = None,
    *,
    verify_owner: bool = True,
) -> Optional[Data]:
    """Retrieve data by ID.

    Every id ever issued keeps resolving. With ``dataset_id`` the lookup is
    precise: exact id first, then the recorded pre-fork ``legacy_id`` within
    that dataset. Without it, an exact match wins ONLY when unambiguous — if
    the id forked during the dataset-scoping upgrade (other rows record it as
    their ``legacy_id``), the lookup refuses with :class:`AmbiguousDataIdError`
    carrying every candidate, instead of silently returning the wrong
    dataset's document. A fork with no surviving exact match and exactly one
    candidate is returned directly (still unambiguous).

    Args:
        user_id (UUID): user ID
        data_id (UUID): ID of the data to retrieve (current or pre-fork)
        dataset_id (Optional[UUID]): dataset scope for precise resolution
        verify_owner (bool): reject a row owned by another user. Pass False
            only when the caller has ALREADY authorized through dataset
            permissions and scoped the lookup to that dataset — row ownership
            is not the access rule there (``datasets.delete_data`` grants a
            collaborator with the dataset ACL access to rows they do not own,
            so a path that also runs on dataset permissions must not be
            stricter).

    Returns:
        Optional[Data]: The requested data object if found, None otherwise

    Raises:
        UnauthorizedDataAccessError: the resolved data belongs to another user.
        AmbiguousDataIdError: context-free lookup of an id that forked into
            several datasets' documents.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        if dataset_id is not None:
            resolved_id = await resolve_data_id(dataset_id, data_id)
            if resolved_id is None:
                return None
            data = await session.get(Data, resolved_id)
            _check_owner(data, user_id, data_id, verify_owner)
            return data

        data = await session.get(Data, data_id)
        _check_owner(data, user_id, data_id, verify_owner)

        forks = (
            await session.execute(
                select(Data, Dataset.name)
                .join(Dataset, Dataset.id == Data.dataset_id, isouter=True)
                .filter(Data.legacy_id == data_id, Data.owner_id == user_id)
            )
        ).all()
        if not forks:
            return data

        if data is None and len(forks) == 1:
            return forks[0][0]

        candidates = []
        if data is not None:
            candidates.append(
                {"dataset_id": data.dataset_id, "dataset_name": None, "data_id": data.id}
            )
        candidates.extend(
            {"dataset_id": fork.dataset_id, "dataset_name": dataset_name, "data_id": fork.id}
            for fork, dataset_name in forks
        )
        raise AmbiguousDataIdError(data_id, candidates)


def _check_owner(
    data: Optional[Data], user_id: UUID, data_id: UUID, verify_owner: bool = True
) -> None:
    if verify_owner and data and data.owner_id != user_id:
        raise UnauthorizedDataAccessError(
            message=f"User {user_id} is not authorized to access data {data_id}"
        )

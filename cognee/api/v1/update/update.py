from uuid import UUID
from typing import Union, BinaryIO, List, Optional

from cognee.modules.users.models import User
from cognee.api.v1.delete import delete
from cognee.api.v1.add import add
from cognee.api.v1.cognify import cognify


async def update(
    data_id: UUID,
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    user: User = None,
    node_set: Optional[List[str]] = None,
    dataset_id: Optional[UUID] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    preferred_loaders: List[str] = None,
    incremental_loading: bool = True,
):
    await delete(
        data_id=data_id,
        dataset_id=dataset_id,
        user=user,
    )

    await add(
        data=data,
        dataset_id=dataset_id,
        user=user,
        node_set=node_set,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        preferred_loaders=preferred_loaders,
        incremental_loading=incremental_loading,
    )

    cognify_run = await cognify(
        datasets=[dataset_id],
        user=user,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
    )

    return cognify_run

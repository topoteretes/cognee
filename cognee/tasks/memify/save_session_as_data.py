from typing import AsyncGenerator, List, Optional
from uuid import UUID

from cognee.modules.data.models import Data
from cognee.modules.users.models import User
from cognee.tasks.ingestion.ingest_data import ingest_data


async def save_session_as_data(
    session_strings: List[str],
    dataset_id: UUID,
    user: User,
    node_set: Optional[List[str]] = None,
) -> AsyncGenerator[Data, None]:
    """Persist session text(s) as Data rows and yield each row.

    Used in place of the older ``cognify_session`` (add + cognify per item)
    pattern: emitting Data rows directly lets the downstream cognify stages
    process every session in one pipeline run, avoiding the dataset-level
    race caused by nested ``cognee.cognify()`` calls.
    """
    if not session_strings:
        return

    data_rows = await ingest_data(
        data=list(session_strings),
        dataset_name=None,
        user=user,
        node_set=node_set,
        dataset_id=dataset_id,
    )

    for data_row in data_rows:
        yield data_row

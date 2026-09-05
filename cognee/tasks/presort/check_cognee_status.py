"""
Incremental-status probe for presort: for every scanned file's content hash,
check whether the same content already exists in any of the user's datasets
and whether it has been cognified there — so a presort report never assumes
the folder is all new and fresh.

Read-only. Scoped to the requesting owner/tenant (the same predicates as
``identify_many``), but across all of the user's datasets rather than one:
other tenants' copies of a file are invisible by design.
"""

from collections import defaultdict
from typing import List

from cognee.shared.logging_utils import get_logger

from .models import FileRecord

logger = get_logger("presort")

_CHUNK_SIZE = 900  # stay under SQLite's bind-parameter limit (see identify_many)

COGNIFY_PIPELINE_NAME = "cognify_pipeline"


async def check_cognee_status(files: List[FileRecord], user=None) -> List[str]:
    """Fill ``cognee_status`` / ``known_in_datasets`` on records; returns warnings.

    Statuses: ``new`` (content unknown to cognee), ``staged`` (added but not
    cognified anywhere), ``cognified`` (processed in at least one dataset).
    Records without a content hash — or every record, if the relational DB is
    unreachable — keep ``unknown``.
    """
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models.Data import Data
    from cognee.modules.data.models.Dataset import Dataset
    from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

    hashes = sorted({record.content_hash for record in files if record.content_hash})
    if not hashes:
        return []

    known_datasets: dict = defaultdict(set)
    cognified_hashes: set = set()

    try:
        if user is None:
            from cognee.modules.users.methods import get_default_user

            user = await get_default_user()

        tenant_id = getattr(user, "tenant_id", None)
        tenant_filter = Data.tenant_id == tenant_id if tenant_id else Data.tenant_id.is_(None)

        async with get_relational_engine().get_async_session() as session:
            for start in range(0, len(hashes), _CHUNK_SIZE):
                chunk = hashes[start : start + _CHUNK_SIZE]
                rows = await session.execute(
                    select(
                        Data.content_hash,
                        Data.pipeline_status,
                        Data.dataset_id,
                        Dataset.name,
                    )
                    .join(Dataset, Data.dataset_id == Dataset.id, isouter=True)
                    .filter(
                        Data.content_hash.in_(chunk),
                        Data.owner_id == user.id,
                        tenant_filter,
                    )
                )
                for content_hash, pipeline_status, dataset_id, dataset_name in rows.fetchall():
                    known_datasets[content_hash].add(dataset_name or str(dataset_id))
                    cognify_status = (pipeline_status or {}).get(COGNIFY_PIPELINE_NAME, {})
                    if (
                        cognify_status.get(str(dataset_id))
                        == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
                    ):
                        cognified_hashes.add(content_hash)
    except Exception as error:
        logger.warning(f"Presort cognee-status check failed: {error}")
        return [
            "cognee-status check failed (relational database unreachable?); "
            "file statuses left as 'unknown'"
        ]

    for record in files:
        if not record.content_hash:
            continue
        if record.content_hash in cognified_hashes:
            record.cognee_status = "cognified"
        elif record.content_hash in known_datasets:
            record.cognee_status = "staged"
        else:
            record.cognee_status = "new"
        record.known_in_datasets = sorted(known_datasets.get(record.content_hash, ()))

    return []

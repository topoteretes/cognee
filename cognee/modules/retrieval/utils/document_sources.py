"""A cited connector chunk's document title and link, carried in the chunk's payload.

A document-mode connector row keeps them in its data record's ``system_metadata`` (``title``,
``url``, written by ``resolve_dlt_sources._build_document_data_item``); chunks carry only the
stored file name. With references asked for, the retrievers that cite chunks copy them into
the payload as ``document_title`` / ``document_url``, exactly as stored; whoever prints them
escapes them for its own format.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.retrieval.utils.results import payload
from cognee.shared.logging_utils import get_logger

logger = get_logger("document_sources")

TITLE_KEY = "document_title"
URL_KEY = "document_url"


async def name_chunk_documents(chunks: list) -> None:
    """Copy each connector chunk's document title and link into its payload, in one read.

    Only the records of the dataset the search runs in are read (``current_dataset_id``); a
    chunk of any other dataset keeps no title. Values that are not text are left out.
    """
    payloads_by_id: dict[UUID, list[dict]] = {}
    for chunk in chunks:
        chunk_payload = payload(chunk)
        if chunk_payload.get("document_id"):
            payloads_by_id.setdefault(UUID(chunk_payload["document_id"]), []).append(chunk_payload)
    if not payloads_by_id:
        return

    statement = select(Data.id, Data.system_metadata).where(Data.id.in_(payloads_by_id))
    dataset_id = current_dataset_id.get()
    if dataset_id is not None:
        statement = statement.where(Data.dataset_id == dataset_id)
    try:
        async with get_relational_engine().get_async_session() as session:
            rows = (await session.execute(statement)).all()
    except SQLAlchemyError as error:
        # A store that cannot be read leaves the chunks unnamed; it never fails the search.
        logger.warning("Unable to read the cited documents' titles: %s", error)
        return

    for data_id, system_metadata in rows:
        metadata = system_metadata or {}
        named = {
            key: metadata.get(field)
            for key, field in ((TITLE_KEY, "title"), (URL_KEY, "url"))
            if isinstance(metadata.get(field), str) and metadata.get(field)
        }
        for chunk_payload in payloads_by_id[data_id]:
            chunk_payload.update(named)

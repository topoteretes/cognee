"""A cited connector chunk's document title and link, carried in the chunk's payload.

A document-mode connector row keeps them in its data record's ``system_metadata`` (``title``,
``url``, written by ``resolve_dlt_sources._build_document_data_item``), with the place it sits
in (``structure``); chunks carry only the stored file name. With references asked for, the
retrievers that cite chunks copy them into the payload as ``document_title`` /
``document_url`` / ``document_parent`` (the nearest ancestor's name), exactly as stored;
whoever prints them escapes them for its own format.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.retrieval.utils.results import payload
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import as_uuid

logger = get_logger("document_sources")

TITLE_KEY = "document_title"
URL_KEY = "document_url"
PARENT_KEY = "document_parent"


async def name_chunk_documents(chunks: list) -> None:
    """Copy each connector chunk's document title and link into its payload, in one read.

    Only the records of the dataset the search runs in are read (``current_dataset_id``); a
    chunk of any other dataset keeps no title. Values that are not text are left out, and a
    chunk whose ``document_id`` is not a data id (a custom chunker's string) is left as is.
    """
    payloads_by_id: dict[UUID, list[dict]] = {}
    for chunk in chunks:
        chunk_payload = payload(chunk)
        data_id = as_uuid(chunk_payload.get("document_id"))
        if data_id is not None:
            payloads_by_id.setdefault(data_id, []).append(chunk_payload)
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
        parent = _parent_name(metadata.get("structure"))
        if parent:
            named[PARENT_KEY] = parent
        for chunk_payload in payloads_by_id[data_id]:
            chunk_payload.update(named)


def _parent_name(structure: object) -> str | None:
    """The name of the nearest place the row sits in (a Slack channel's ``#name``, a Notion
    page), from the structure its connector wrote; ``None`` when it names none."""
    ancestors = structure.get("ancestors") if isinstance(structure, dict) else None
    nearest = ancestors[0] if isinstance(ancestors, list) and ancestors else None
    name = nearest.get("name") if isinstance(nearest, dict) else None
    return name if isinstance(name, str) and name else None

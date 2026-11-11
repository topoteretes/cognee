"""The title and link of a cited connector document, carried in its chunks' payloads.

A document-mode connector row (``resolve_dlt_sources._build_document_data_item``) keeps them
in its data record's ``system_metadata`` (``title``, ``url``); chunks carry only the stored
file name (``text_<md5>``). The retrievers that cite chunks (RAG, hybrid, temporal) name them
where they retrieve them, when references are asked for, as ``document_title`` and
``document_url`` in the payload a verbose search returns. Nothing else changes: the answer, its
Evidence text and the structured evidence stay as they are, and every other document's payload
gains nothing (``Data.label`` is not read, so a label given to ``add()`` changes nothing here).

``system_metadata`` is not an ``add()`` input over HTTP; an SDK caller who sets those keys on a
``DataItem`` opts in, and a title that is not text is left out. Both are provider content, so
the title is kept on one line and the link only when it is a plain http(s) URL.
"""

import unicodedata
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.shared.logging_utils import get_logger

logger = get_logger("document_sources")

# Payload keys a connector chunk carries its document's title and link under, once named;
# absent on every other chunk.
TITLE_KEY = "document_title"
URL_KEY = "document_url"


@dataclass(frozen=True)
class DocumentSource:
    title: str | None
    url: str | None


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _printable_url(url: Any) -> bool:
    """A web link that cannot end a ``<url>`` it is printed in or start a new line."""
    return (
        isinstance(url, str)
        and url.startswith(("https://", "http://"))
        and not any(
            char.isspace() or char in '<>|"' or unicodedata.category(char) == "Cf" for char in url
        )
    )


def document_source(system_metadata: dict | None) -> DocumentSource:
    """The title and link of a connector row, safe to print."""
    metadata = system_metadata or {}
    title, url = metadata.get("title"), metadata.get("url")
    return DocumentSource(
        title=(_one_line(title) or None) if isinstance(title, str) else None,
        url=url if _printable_url(url) else None,
    )


def _payload(chunk: Any) -> dict:
    if isinstance(chunk, dict):
        inner = chunk.get("payload")
        return inner if isinstance(inner, dict) else chunk
    payload = getattr(chunk, "payload", None)
    return payload if isinstance(payload, dict) else {}


async def name_chunk_documents(chunks: list, dataset_id: UUID | None) -> None:
    """Write each connector chunk's document title and link into its payload, from one read.

    ``dataset_id`` is the dataset the search named (``None`` only for a search without one), so
    only its records are read; a chunk of another dataset, possible in a store shared with
    backend access control off, keeps its file name.
    """
    if dataset_id is not None and not isinstance(dataset_id, UUID):
        raise TypeError(f"dataset_id must be a UUID, not {type(dataset_id).__name__}")

    payloads_by_id: dict[UUID, list[dict]] = {}
    for chunk in chunks:
        payload = _payload(chunk)
        if payload.get("document_id"):
            payloads_by_id.setdefault(UUID(payload["document_id"]), []).append(payload)
    if not payloads_by_id:
        return

    statement = select(Data.id, Data.system_metadata).where(Data.id.in_(payloads_by_id))
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
        source = document_source(system_metadata)
        for payload in payloads_by_id[data_id]:
            if source.title:
                payload[TITLE_KEY] = source.title
            if source.url:
                payload[URL_KEY] = source.url

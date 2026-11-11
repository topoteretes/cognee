"""The title and link a reference names its document by.

They live only on the document's data record (``Data.label``, ``Data.system_metadata["url"]``);
chunks carry the stored file name (``text_<md5>``). Both may come from a provider's content, so
they are made safe to print on one line before any reference shows them.
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

# Payload keys a cited chunk carries its document's title and link under, once named.
TITLE_KEY = "document_title"
URL_KEY = "document_url"


@dataclass(frozen=True)
class DocumentSource:
    title: str | None
    url: str | None


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _printable_url(url: Any) -> bool:
    """A web link that cannot end the ``<url>`` it is printed in or start a new line."""
    return (
        isinstance(url, str)
        and url.startswith(("https://", "http://"))
        and not any(
            char.isspace() or char in '<>|"' or unicodedata.category(char) == "Cf" for char in url
        )
    )


def document_source(label: str | None, system_metadata: dict | None) -> DocumentSource:
    """The title and link of a data record, as a reference may print them."""
    url = (system_metadata or {}).get("url")
    return DocumentSource(
        title=_one_line(label) or None if label else None,
        url=url if _printable_url(url) else None,
    )


def document_label(document_name: str | None, title: str | None, url: str | None) -> str:
    """How an Evidence bullet names a document: ``"<title>" <url>``, else its file."""
    name = (
        f'"{title.replace(chr(34), chr(39))}"'
        if title
        else f"document {_one_line(document_name or '') or 'unknown'}"
    )
    return f"{name} <{url}>" if url else name


def _payload(chunk: Any) -> dict | None:
    if isinstance(chunk, dict):
        inner = chunk.get("payload")
        return inner if isinstance(inner, dict) else chunk
    payload = getattr(chunk, "payload", None)
    return payload if isinstance(payload, dict) else None


async def name_chunk_documents(chunks: list, dataset_id: UUID | None) -> None:
    """Write each chunk's document title and link into its payload, from one read.

    ``dataset_id`` is the dataset the search read, so nothing outside it is read; it is
    ``None`` only with backend access control off, where every dataset shares one store.
    """
    if dataset_id is not None and not isinstance(dataset_id, UUID):
        raise TypeError(f"dataset_id must be a UUID, not {type(dataset_id).__name__}")

    payloads_by_id: dict[UUID, list[dict]] = {}
    for chunk in chunks:
        payload = _payload(chunk)
        if payload and payload.get("document_id"):
            payloads_by_id.setdefault(UUID(payload["document_id"]), []).append(payload)
    if not payloads_by_id:
        return

    statement = select(Data.id, Data.label, Data.system_metadata).where(Data.id.in_(payloads_by_id))
    if dataset_id is not None:
        statement = statement.where(Data.dataset_id == dataset_id)
    try:
        async with get_relational_engine().get_async_session() as session:
            rows = (await session.execute(statement)).all()
    except SQLAlchemyError as error:
        # A reference never fails the answer: the documents keep their file names.
        logger.warning("Unable to read the cited documents' titles: %s", error)
        return

    for data_id, label, system_metadata in rows:
        source = document_source(label, system_metadata)
        for payload in payloads_by_id[data_id]:
            payload[TITLE_KEY] = source.title
            payload[URL_KEY] = source.url

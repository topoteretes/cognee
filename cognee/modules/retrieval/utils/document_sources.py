"""The title and link of the documents a search cites, read from their data records.

A connector row keeps its title in ``Data.label`` and its link in
``Data.system_metadata["url"]``; a plain ``add`` keeps the caller's ``label``. Neither
reaches the chunks, which carry only the stored file name (``text_<md5>`` for text), so a
reference reads them here by the ``data_id`` its chunk carries. The data record is the one
place they live, so a reference always shows the document's current title.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from cognee.modules.search.models.EvidenceReference import EvidenceReference
from cognee.shared.logging_utils import get_logger

logger = get_logger("document_sources")


@dataclass(frozen=True)
class DocumentSource:
    """What a reference names a document by; either may be missing."""

    title: str | None
    url: str | None


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _uuid(value: Any) -> UUID | None:
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def document_label(document_name: str | None, source: DocumentSource | None) -> str:
    """How an Evidence bullet names its document: ``"<title>" <url>`` when the document
    has them, else ``document <file name>`` as before."""
    title = source.title if source else None
    url = source.url if source else None
    name = f'"{title}"' if title else f"document {document_name or 'unknown'}"
    return f"{name} <{url}>" if url else name


async def get_document_sources(
    data_ids: Iterable[Any], dataset_id: Any = None
) -> dict[str, DocumentSource]:
    """The title and link of each cited document that has either, keyed by data id.

    One query over the data records of the ids. ``dataset_id`` is the dataset the search
    read: rows are dataset-scoped, so nothing outside it is read. It is ``None`` only for
    a search without a dataset (backend access control off, one shared store), where the
    ids are those of chunks this search already returned. Like every reference helper it
    never fails the answer: a lookup that fails leaves the documents named by their file.
    """
    ids = {parsed for data_id in data_ids if (parsed := _uuid(data_id)) is not None}
    if not ids:
        return {}

    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    statement = select(Data.id, Data.label, Data.system_metadata).where(Data.id.in_(ids))
    scope = _uuid(dataset_id) if dataset_id is not None else None
    if scope is not None:
        statement = statement.where(Data.dataset_id == scope)
    try:
        async with get_relational_engine().get_async_session() as session:
            rows = (await session.execute(statement)).all()
    except Exception as error:
        logger.debug("Unable to read the cited documents' titles: %s", error, exc_info=True)
        return {}

    sources: dict[str, DocumentSource] = {}
    for data_id, label, system_metadata in rows:
        metadata = system_metadata if isinstance(system_metadata, dict) else {}
        source = DocumentSource(title=_text(label), url=_text(metadata.get("url")))
        if source.title or source.url:
            sources[str(data_id)] = source
    return sources


async def name_evidence_sources(
    evidence: list[EvidenceReference], dataset_id: Any = None
) -> list[EvidenceReference]:
    """``evidence`` with each reference's document title and link filled in."""
    sources = await get_document_sources(
        (reference.data_id for reference in evidence if reference.data_id), dataset_id
    )
    if not sources:
        return evidence
    named = []
    for reference in evidence:
        source = sources.get(reference.data_id or "")
        named.append(
            reference.model_copy(update={"title": source.title, "url": source.url})
            if source
            else reference
        )
    return named

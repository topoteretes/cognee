"""Find the stored documents an input came from, by origin rather than by content.

Content-hash dedup (``identify_many``) answers "is this exact content already
stored?". This answers the other question an edited file raises: "which stored
document is this a new version of?". The answer is the input's *origin*: the
source path of a local file (``Data.original_data_location``), or the filename
of an upload, whose stored location carries a content hash and so cannot be
matched. Raw text has neither — it is named by its content hash, so an edited
text is a new document by construction and can only be updated by ``data_id``.

The lookup is one batched query in the dedup scope, (dataset, owner, tenant):
another user's same-named file is never a match. Both ``add()`` (to refuse an
update in disguise) and ``update()`` (to infer the document a replacement
targets) resolve origins through here, so the two agree on what "the same
document" means.

Not matched: s3:// objects, URLs and DLT sources, whose bytes are not local.
"""

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname
from uuid import UUID

from sqlalchemy import or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path
from cognee.modules.data.models.Data import Data
from cognee.modules.users.models import User


@dataclass(frozen=True)
class Origin:
    """How an input names the stored document it is a version of.

    Exactly one of ``location`` (a file URI, matched on
    ``Data.original_data_location``) and ``name`` (an upload's ``(stem,
    extension)``, matched on ``Data.name`` / ``Data.extension``) is set.
    ``label`` is the input as a person would name it, for messages.
    """

    label: str
    content_hash: str
    location: str | None = None
    name: tuple[str, str] | None = None


@dataclass(frozen=True)
class OriginMatch:
    """A stored document with the same origin as an input."""

    data_id: UUID
    content_hash: str


def _local_path(item: Any) -> Path | None:
    """The existing local path ``item`` names (a path or a file:// URI), or None."""
    if not isinstance(item, (str, Path)):
        return None
    if isinstance(item, str) and item.startswith("file://"):
        item = url2pathname(urlparse(item).path)
    try:
        return resolve_local_path(item, must_exist=True)
    except (FileNotFoundError, OSError, ValueError):
        return None


def expand_directories(items: list[Any]) -> list[Any]:
    """``items`` with every local directory replaced by the files under it, in path order."""
    expanded: list[Any] = []
    for item in items:
        path = _local_path(item)
        if path is not None and path.is_dir():
            expanded.extend(str(p) for p in sorted(path.rglob("*")) if p.is_file())
        else:
            expanded.append(item)
    return expanded


async def origin_of(item: Any) -> Origin | None:
    """The origin of one input, or None when the input has none.

    A local file path is matched by location. An upload (an object with
    ``file`` and ``filename``, as FastAPI hands them in) is matched by
    filename: the stem is the stored row's name and the extension comes from
    the content type guessed with the filename as a hint, both derived the way
    ingestion derives them.
    """
    path = _local_path(item)
    if path is not None and path.is_file():
        with open(path, "rb") as file:
            metadata = await get_file_metadata(file, name=path.name)
        return Origin(
            label=path.name, content_hash=metadata["content_hash"], location=path.as_uri()
        )

    stream = getattr(item, "file", None)
    filename = getattr(item, "filename", None)
    if stream is None or not filename:
        return None

    metadata = await get_file_metadata(stream, name=str(filename))
    stream.seek(0)
    stem = PureWindowsPath(str(filename)).stem
    return Origin(
        label=str(filename),
        content_hash=metadata["content_hash"],
        name=(stem, metadata["extension"]),
    )


async def find_by_origin(
    origins: list[Origin], user: User, dataset_id: UUID
) -> dict[Origin, list[OriginMatch]]:
    """The stored documents in ``dataset_id`` sharing an origin with each of ``origins``.

    One query for the whole batch. Every origin is a key of the result, with an
    empty list when nothing in the dataset came from it.
    """
    matches: dict[Origin, list[OriginMatch]] = {origin: [] for origin in origins}
    if not origins:
        return matches

    by_location = {origin.location: origin for origin in origins if origin.location}
    by_name = {origin.name: origin for origin in origins if origin.name}
    tenant_filter = Data.tenant_id == user.tenant_id if user.tenant_id else Data.tenant_id.is_(None)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (
            await session.execute(
                select(
                    Data.id,
                    Data.name,
                    Data.extension,
                    Data.original_data_location,
                    Data.content_hash,
                ).filter(
                    Data.dataset_id == dataset_id,
                    Data.owner_id == user.id,
                    tenant_filter,
                    or_(
                        Data.original_data_location.in_(list(by_location)),
                        Data.name.in_([stem for stem, _ in by_name]),
                    ),
                )
            )
        ).fetchall()

    for data_id, name, extension, location, content_hash in rows:
        origin = by_location.get(location) or by_name.get((name, extension))
        if origin is not None:
            matches[origin].append(OriginMatch(data_id=data_id, content_hash=str(content_hash)))
    return matches


__all__ = ["Origin", "OriginMatch", "expand_directories", "find_by_origin", "origin_of"]

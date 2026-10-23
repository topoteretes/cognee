import hashlib
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import BinaryIO

from .IngestionData import IngestionData

# Long enough for any title a connector hands over, short enough for every
# object store and filesystem cognee writes to (255-byte basenames). A byte
# budget, not a character count: a CJK or emoji title is 3-4 bytes per char.
_MAX_TEXT_NAME_BYTES = 200
_TEXT_SUFFIX = ".txt"


def text_file_name(name: str | None) -> str | None:
    """The ``.txt`` basename a caller-named text payload is stored under.

    Takes a document title or filename as the caller knows it and returns a
    safe object basename: directory components are dropped (a name never
    chooses where in storage it lands), control characters and leading dots
    are stripped, and ``.txt`` is appended when the name does not already end
    in it. The suffix is appended rather than swapped in, so ``roadmap.docx``
    (a Drive export whose *text* this is) becomes ``roadmap.docx.txt``: the
    stored object is read back as text, while the document name a user sees,
    the stem, stays ``roadmap.docx``. Returns ``None`` for an unusable name, so
    the caller falls back to the content-hash name.
    """
    if not name:
        return None
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip().lstrip(".").rstrip(". ")
    if not base:
        return None
    if base.lower().endswith(_TEXT_SUFFIX):
        base = base[: -len(_TEXT_SUFFIX)].rstrip(". ")
        if not base:
            return None
    budget = _MAX_TEXT_NAME_BYTES - len(_TEXT_SUFFIX)
    if len(base.encode("utf-8")) > budget:
        # Cut on the byte budget without splitting a code point.
        base = base.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip(". ")
    return f"{base}{_TEXT_SUFFIX}"


def create_text_data(data: str, name: str | None = None) -> "TextData":
    return TextData(data, name=name)


class TextData(IngestionData):
    data: str = None
    metadata: dict = None
    # The caller's document name as a ``.txt`` basename, or None for text
    # named by its content hash (``text_<md5>.txt``).
    name: str | None = None

    def __init__(self, data: BinaryIO, name: str | None = None) -> None:
        self.data = data
        self.name = text_file_name(name)

    def get_identifier(self) -> str:
        metadata = self.get_metadata()

        return metadata["content_hash"]

    def get_metadata(self) -> dict:
        self.ensure_metadata()

        return self.metadata

    async def aget_identifier(self) -> str:
        metadata = await self.aget_metadata()

        return metadata["content_hash"]

    async def aget_metadata(self) -> dict:
        # Text is already in memory: no I/O to move off the event loop.
        self.ensure_metadata()

        return self.metadata

    def ensure_metadata(self) -> None:
        if self.metadata is None:
            self.metadata = {}

        data_contents = self.data.encode("utf-8")
        hash_contents = hashlib.md5(data_contents).hexdigest()
        self.metadata["name"] = self.name or ("text_" + hash_contents + ".txt")
        self.metadata["content_hash"] = hash_contents
        # Describe the payload the same way reading the stored ".txt" back would
        # (``guess_file_type`` resolves a .txt extension to exactly this type).
        # Ingestion builds ``Data`` rows straight from this dict, so a partial
        # one would drop the row's extension, mime type and size.
        self.metadata["mime_type"] = "text/plain"
        self.metadata["extension"] = "txt"
        self.metadata["file_size"] = len(data_contents)
        self.metadata.setdefault("file_path", None)

    @asynccontextmanager
    async def get_data(self) -> AsyncGenerator[str, None]:
        yield self.data

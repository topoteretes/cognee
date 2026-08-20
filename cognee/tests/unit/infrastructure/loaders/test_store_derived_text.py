"""A loader describing the text it wrote must match reading that text back.

``ingest_data`` builds the ``Data`` row's extension, mime type, size and
``raw_content_hash`` from the derived-text file. It used to learn them by
re-opening the file the loader had just written — over S3, a HEAD plus a full
GET of content the loader still had in memory. Loaders now describe their own
output instead, which is only safe while the two agree exactly. These tests pin
that equivalence.
"""

import pytest

import cognee.modules.ingestion as ingestion
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.loaders.LoaderInterface import LoaderResult
from cognee.infrastructure.loaders.store_derived_text import store_derived_text

CONTENTS = [
    pytest.param("plain ascii text\nsecond line\n", id="ascii"),
    pytest.param("unicode: naïve café — 日本語 🎉\n", id="unicode"),
    pytest.param("carriage\r\nreturns\r\nkept\r\n", id="crlf"),
    pytest.param("", id="empty"),
    pytest.param("x" * 200_000, id="large"),
]


async def _read_back_metadata(file_path: str):
    """Describe a stored file the way ingestion used to: open it and classify."""
    async with open_data_file(file_path) as file:
        return await ingestion.classify(file).aget_metadata()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", CONTENTS)
async def test_description_matches_reading_the_file_back(tmp_path, content):
    storage = LocalFileStorage(str(tmp_path))

    result = await store_derived_text(storage, "text_derived.txt", content)

    assert isinstance(result, LoaderResult)
    assert result.file_metadata == await _read_back_metadata(result.file_path)


@pytest.mark.asyncio
async def test_content_hash_is_the_digest_of_the_bytes_on_disk(tmp_path):
    # Both storage backends write str payloads as UTF-8 with newline="\n" (no
    # translation), so the digest computed in memory must equal the digest of
    # the file as written — including for content that contains "\r\n".
    content = "alpha\r\nbeta\n"
    storage = LocalFileStorage(str(tmp_path))

    result = await store_derived_text(storage, "text_newlines.txt", content)

    on_disk = (tmp_path / "text_newlines.txt").read_bytes()
    assert on_disk == content.encode("utf-8")
    assert result.file_metadata["file_size"] == len(on_disk)


@pytest.mark.asyncio
async def test_extra_loader_result_fields_are_carried_through(tmp_path):
    # Loaders that also own the record's identity/route stamp (dlt, code) pass
    # those through the same helper.
    storage = LocalFileStorage(str(tmp_path))

    result = await store_derived_text(
        storage, "text_stamped.txt", "body", system_metadata={"source": "code"}
    )

    assert result.system_metadata == {"source": "code"}
    assert result.file_metadata is not None

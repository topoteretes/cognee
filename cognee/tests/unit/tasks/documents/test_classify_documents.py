import uuid
from types import SimpleNamespace

import pytest

from cognee.modules.data.processing.document_types import (
    CsvDocument,
    PdfDocument,
    TextDocument,
)
from cognee.tasks.documents.classify_documents import classify_documents


def _make_data_item(extension):
    return SimpleNamespace(
        id=uuid.uuid4(),
        name="sample",
        extension=extension,
        mime_type="text/plain",
        raw_data_location="/tmp/sample",
        external_metadata={},
        importance_weight=0.5,
    )


@pytest.mark.parametrize(
    "extension,expected_class",
    [
        ("md", TextDocument),
        ("json", TextDocument),
        ("yaml", TextDocument),
        ("xml", TextDocument),
        ("csv", CsvDocument),
        ("pdf", PdfDocument),
        ("PDF", PdfDocument),  # case-insensitive lookup
        ("unknown_ext", TextDocument),  # falls back instead of raising KeyError
    ],
)
@pytest.mark.asyncio
async def test_classify_documents_resolves_extensions(extension, expected_class):
    documents = await classify_documents([_make_data_item(extension)])

    assert len(documents) == 1
    assert isinstance(documents[0], expected_class)

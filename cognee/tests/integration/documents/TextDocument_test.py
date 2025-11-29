import os
import sys
import uuid
import pytest
import pathlib
from unittest.mock import patch

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.tests.integration.documents.AudioDocument_test import mock_get_embedding_engine
from cognee.tests.integration.documents.async_gen_zip import async_gen_zip

chunk_by_sentence_module = sys.modules.get("cognee.tasks.chunks.chunk_by_sentence")


GROUND_TRUTH = {
    "code.txt": [
        {"word_count": 252, "len_text": 1376, "cut_type": "paragraph_end"},
        {"word_count": 56, "len_text": 481, "cut_type": "paragraph_end"},
    ],
    "Natural_language_processing.txt": [
        {"word_count": 128, "len_text": 984, "cut_type": "paragraph_end"},
        {"word_count": 1, "len_text": 1, "cut_type": "paragraph_end"},
    ],
}


@pytest.mark.parametrize(
    "input_file,chunk_size,importance_weight,expected_weight",
    [
        ("code.txt", 256, 0.9, 0.9),
        ("Natural_language_processing.txt", 128, None, 0.5),
    ],
)
@patch.object(
    chunk_by_sentence_module, "get_embedding_engine", side_effect=mock_get_embedding_engine
)
@pytest.mark.asyncio
async def test_TextDocument(mock_engine, input_file, chunk_size, expected_weight):
    test_file_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent, "test_data", input_file
    )
    document = TextDocument(
        id=uuid.uuid4(),
        name=input_file,
        raw_data_location=test_file_path,
        external_metadata="",
        mime_type="",
        importance_weight=importance_weight,
    )

    async for ground_truth, paragraph_data in async_gen_zip(
            GROUND_TRUTH[input_file],
            document.read(chunker_cls=TextChunker, max_chunk_size=chunk_size),
    ):
        assert ground_truth["word_count"] == paragraph_data.chunk_size, (
            f'{ground_truth["word_count"] = } != {paragraph_data.chunk_size = }'
        )
        assert ground_truth["len_text"] == len(paragraph_data.text), (
            f'{ground_truth["len_text"] = } != {len(paragraph_data.text) = }'
        )
        assert ground_truth["cut_type"] == paragraph_data.cut_type, (
            f'{ground_truth["cut_type"] = } != {paragraph_data.cut_type = }'
        )

        assert hasattr(paragraph_data, "importance_weight"), (
            "DocumentChunk object is missing the 'importance_weight' attribute."
        )
        assert paragraph_data.importance_weight == expected_weight, (
            f"Chunk importance_weight failed for Document {input_file}. "
            f"Expected {expected_weight}, but got {paragraph_data.importance_weight}."
        )
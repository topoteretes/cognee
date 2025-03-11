import os
import uuid

import pytest
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from unittest.mock import patch
from cognee.tests.integration.documents.AudioDocument_test import mock_get_embedding_engine
import sys

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
    "input_file,chunk_size",
    [("code.txt", 256), ("Natural_language_processing.txt", 128)],
)
@patch.object(
    chunk_by_sentence_module, "get_embedding_engine", side_effect=mock_get_embedding_engine
)
def test_TextDocument(mock_engine, input_file, chunk_size):
    test_file_path = os.path.join(
        os.sep,
        *(os.path.dirname(__file__).split(os.sep)[:-2]),
        "test_data",
        input_file,
    )
    document = TextDocument(
        id=uuid.uuid4(),
        name=input_file,
        raw_data_location=test_file_path,
        external_metadata="",
        mime_type="",
    )

    for ground_truth, paragraph_data in zip(
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

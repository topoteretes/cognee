import os
import uuid
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.data.processing.document_types.PdfDocument import PdfDocument
from cognee.tests.integration.documents.AudioDocument_test import mock_get_embedding_engine
from unittest.mock import patch
import sys

chunk_by_sentence_module = sys.modules.get("cognee.tasks.chunks.chunk_by_sentence")


GROUND_TRUTH = [
    {"word_count": 879, "len_text": 5697, "cut_type": "sentence_end"},
    {"word_count": 953, "len_text": 6473, "cut_type": "sentence_end"},
]


@patch.object(
    chunk_by_sentence_module, "get_embedding_engine", side_effect=mock_get_embedding_engine
)
def test_PdfDocument(mock_engine):
    test_file_path = os.path.join(
        os.sep,
        *(os.path.dirname(__file__).split(os.sep)[:-2]),
        "test_data",
        "artificial-intelligence.pdf",
    )
    document = PdfDocument(
        id=uuid.uuid4(),
        name="Test document.pdf",
        raw_data_location=test_file_path,
        external_metadata="",
        mime_type="",
    )

    for ground_truth, paragraph_data in zip(
        GROUND_TRUTH, document.read(chunker_cls=TextChunker, max_chunk_size=1024)
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

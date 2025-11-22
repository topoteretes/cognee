import os
import sys
import uuid
import pytest
import pathlib
from unittest.mock import patch

from cognee.modules.chunking.CsvChunker import CsvChunker
from cognee.modules.data.processing.document_types.CsvDocument import CsvDocument
from cognee.tests.integration.documents.AudioDocument_test import mock_get_embedding_engine
from cognee.tests.integration.documents.async_gen_zip import async_gen_zip

chunk_by_row_module = sys.modules.get("cognee.tasks.chunks.chunk_by_row")


GROUND_TRUTH = {
    "chunk_size_10": [
        {"token_count": 9, "len_text": 26, "cut_type": "row_cut", "chunk_index": 0},
        {"token_count": 6, "len_text": 29, "cut_type": "row_end", "chunk_index": 1},
        {"token_count": 9, "len_text": 25, "cut_type": "row_cut", "chunk_index": 2},
        {"token_count": 6, "len_text": 30, "cut_type": "row_end", "chunk_index": 3},
    ],
    "chunk_size_128": [
        {"token_count": 15, "len_text": 57, "cut_type": "row_end", "chunk_index": 0},
        {"token_count": 15, "len_text": 57, "cut_type": "row_end", "chunk_index": 1},
    ],
}


@pytest.mark.parametrize(
    "input_file,chunk_size",
    [("example_with_header.csv", 10), ("example_with_header.csv", 128)],
)
@patch.object(chunk_by_row_module, "get_embedding_engine", side_effect=mock_get_embedding_engine)
@pytest.mark.asyncio
async def test_CsvDocument(mock_engine, input_file, chunk_size):
    # Define file paths of test data
    csv_file_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent,
        "test_data",
        input_file,
    )

    # Define test documents
    csv_document = CsvDocument(
        id=uuid.uuid4(),
        name="example_with_header.csv",
        raw_data_location=csv_file_path,
        external_metadata="",
        mime_type="text/csv",
    )

    # TEST CSV
    ground_truth_key = f"chunk_size_{chunk_size}"
    async for ground_truth, row_data in async_gen_zip(
        GROUND_TRUTH[ground_truth_key],
        csv_document.read(chunker_cls=CsvChunker, max_chunk_size=chunk_size),
    ):
        assert ground_truth["token_count"] == row_data.chunk_size, (
            f'{ground_truth["token_count"] = } != {row_data.chunk_size = }'
        )
        assert ground_truth["len_text"] == len(row_data.text), (
            f'{ground_truth["len_text"] = } != {len(row_data.text) = }'
        )
        assert ground_truth["cut_type"] == row_data.cut_type, (
            f'{ground_truth["cut_type"] = } != {row_data.cut_type = }'
        )
        assert ground_truth["chunk_index"] == row_data.chunk_index, (
            f'{ground_truth["chunk_index"] = } != {row_data.chunk_index = }'
        )

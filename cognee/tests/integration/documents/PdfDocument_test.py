import os
import uuid

from cognee.modules.data.processing.document_types.PdfDocument import PdfDocument

GROUND_TRUTH = [
    {"word_count": 879, "len_text": 5607, "cut_type": "sentence_end"},
    {"word_count": 953, "len_text": 6363, "cut_type": "sentence_end"},
]


def test_PdfDocument():
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
        metadata_id=uuid.uuid4(),
        mime_type="",
    )

    for ground_truth, paragraph_data in zip(
        GROUND_TRUTH, document.read(chunk_size=1024, chunker="text_chunker")
    ):
        assert ground_truth["word_count"] == paragraph_data.word_count, (
            f'{ground_truth["word_count"] = } != {paragraph_data.word_count = }'
        )
        assert ground_truth["len_text"] == len(paragraph_data.text), (
            f'{ground_truth["len_text"] = } != {len(paragraph_data.text) = }'
        )
        assert ground_truth["cut_type"] == paragraph_data.cut_type, (
            f'{ground_truth["cut_type"] = } != {paragraph_data.cut_type = }'
        )

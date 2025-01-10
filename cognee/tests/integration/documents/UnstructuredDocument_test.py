import os
import uuid

from cognee.modules.data.processing.document_types.UnstructuredDocument import UnstructuredDocument


def test_UnstructuredDocument():
    # Define file paths of test data
    pptx_file_path = os.path.join(
        os.sep,
        *(os.path.dirname(__file__).split(os.sep)[:-2]),
        "test_data",
        "example.pptx",
    )

    docx_file_path = os.path.join(
        os.sep,
        *(os.path.dirname(__file__).split(os.sep)[:-2]),
        "test_data",
        "example.docx",
    )

    csv_file_path = os.path.join(
        os.sep,
        *(os.path.dirname(__file__).split(os.sep)[:-2]),
        "test_data",
        "example.csv",
    )

    xlsx_file_path = os.path.join(
        os.sep,
        *(os.path.dirname(__file__).split(os.sep)[:-2]),
        "test_data",
        "example.xlsx",
    )

    # Define test documents
    pptx_document = UnstructuredDocument(
        id=uuid.uuid4(),
        name="example.pptx",
        raw_data_location=pptx_file_path,
        metadata_id=uuid.uuid4(),
        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    docx_document = UnstructuredDocument(
        id=uuid.uuid4(),
        name="example.docx",
        raw_data_location=docx_file_path,
        metadata_id=uuid.uuid4(),
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    csv_document = UnstructuredDocument(
        id=uuid.uuid4(),
        name="example.csv",
        raw_data_location=csv_file_path,
        metadata_id=uuid.uuid4(),
        mime_type="text/csv",
    )

    xlsx_document = UnstructuredDocument(
        id=uuid.uuid4(),
        name="example.xlsx",
        raw_data_location=xlsx_file_path,
        metadata_id=uuid.uuid4(),
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Test PPTX
    for paragraph_data in pptx_document.read(chunk_size=1024, chunker="text_chunker"):
        assert 19 == paragraph_data.word_count, f" 19 != {paragraph_data.word_count = }"
        assert 104 == len(paragraph_data.text), f" 104 != {len(paragraph_data.text) = }"
        assert "sentence_cut" == paragraph_data.cut_type, (
            f" sentence_cut != {paragraph_data.cut_type = }"
        )

    # Test DOCX
    for paragraph_data in docx_document.read(chunk_size=1024, chunker="text_chunker"):
        assert 16 == paragraph_data.word_count, f" 16 != {paragraph_data.word_count = }"
        assert 145 == len(paragraph_data.text), f" 145 != {len(paragraph_data.text) = }"
        assert "sentence_end" == paragraph_data.cut_type, (
            f" sentence_end != {paragraph_data.cut_type = }"
        )

    # TEST CSV
    for paragraph_data in csv_document.read(chunk_size=1024, chunker="text_chunker"):
        assert 15 == paragraph_data.word_count, f" 15 != {paragraph_data.word_count = }"
        assert "A A A A A A A A A,A A A A A A,A A" == paragraph_data.text, (
            f"Read text doesn't match expected text: {paragraph_data.text}"
        )
        assert "sentence_cut" == paragraph_data.cut_type, (
            f" sentence_cut != {paragraph_data.cut_type = }"
        )

    # Test XLSX
    for paragraph_data in xlsx_document.read(chunk_size=1024, chunker="text_chunker"):
        assert 36 == paragraph_data.word_count, f" 36 != {paragraph_data.word_count = }"
        assert 171 == len(paragraph_data.text), f" 171 != {len(paragraph_data.text) = }"
        assert "sentence_cut" == paragraph_data.cut_type, (
            f" sentence_cut != {paragraph_data.cut_type = }"
        )

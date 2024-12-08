import os
import uuid

from cognee.modules.data.processing.document_types.UnstructuredDocument import UnstructuredDocument

def test_UnstructuredDocument():
    docx_file_path = os.path.join(
        os.sep,
        *(os.path.dirname(__file__).split(os.sep)[:-2]),
        "test_data",
        "example.pptx",
    )

    pptx_document = UnstructuredDocument(
        id=uuid.uuid4(), name="example.pptx", raw_data_location=docx_file_path, metadata_id=uuid.uuid4(),
        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

    for paragraph_data in pptx_document.read(chunk_size=1024):
        assert 19 == paragraph_data.word_count, f' 19 != {paragraph_data.word_count = }'
        assert 104 == len(paragraph_data.text), f' 104 != {len(paragraph_data.text) = }'
        assert 'sentence_cut' == paragraph_data.cut_type, f' sentence_cut != {paragraph_data.cut_type = }'

import uuid
from unittest.mock import patch
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.data.processing.document_types.AudioDocument import AudioDocument
import sys

chunk_by_sentence_module = sys.modules.get("cognee.tasks.chunks.chunk_by_sentence")


def mock_get_embedding_engine():
    class MockEngine:
        tokenizer = None

    return MockEngine()


GROUND_TRUTH = [
    {"word_count": 57, "len_text": 353, "cut_type": "sentence_end"},
    {"word_count": 58, "len_text": 358, "cut_type": "sentence_end"},
    {"word_count": 41, "len_text": 219, "cut_type": "sentence_end"},
]

TEST_TEXT = """
"Mike, we need to talk about the payment processing service."
"Good timing. The board wants one-click checkout by end of quarter."
"That's exactly the problem. The service is held together with duct tape. One wrong move and—"
"Sarah, we've been over this. The market won't wait."
"And neither will a system collapse! The technical debt is crushing us. Every new feature takes twice as long as it should."
"Then work twice as hard. Our competitors—"
"Our competitors will laugh when our whole system goes down during Black Friday! We're talking about financial transactions here, not some blog comments section."
"Write up your concerns in a doc. Right now, we ship one-click."
"Then you'll ship it without me. I won't stake my reputation on a house of cards."
"Are you threatening to quit?"
"No, I'm threatening to be right. And when it breaks, I want it in writing that you chose this."
"The feature ships, Sarah. That's final.\""""


@patch.object(
    chunk_by_sentence_module, "get_embedding_engine", side_effect=mock_get_embedding_engine
)
def test_AudioDocument(mock_engine):
    document = AudioDocument(
        id=uuid.uuid4(),
        name="audio-dummy-test",
        raw_data_location="",
        external_metadata="",
        mime_type="",
    )
    with patch.object(AudioDocument, "create_transcript", return_value=TEST_TEXT):
        for ground_truth, paragraph_data in zip(
            GROUND_TRUTH,
            document.read(chunker_cls=TextChunker, max_chunk_size=64),
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

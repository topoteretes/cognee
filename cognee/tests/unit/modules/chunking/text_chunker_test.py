import uuid
from unittest.mock import MagicMock, patch

import pytest

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document


class TestTextChunker:
    @pytest.fixture
    def mock_document(self):
        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        return doc

    @pytest.fixture
    def mock_get_text(self):
        return lambda: ["This is a test paragraph."]

    def test_check_word_count_and_token_count(self, mock_document, mock_get_text):
        # Setup
        chunker = TextChunker(
            document=mock_document,
            get_text=mock_get_text,
            max_chunk_tokens=100,
            chunk_size=50
        )
        
        # Test when both word count and token count fit
        chunk_data = {"word_count": 10, "token_count": 20}
        result = chunker.check_word_count_and_token_count(20, 30, chunk_data)
        assert result is True
        
        # Test when word count exceeds limit
        chunk_data = {"word_count": 40, "token_count": 20}
        result = chunker.check_word_count_and_token_count(20, 30, chunk_data)
        assert result is False
        
        # Test when token count exceeds limit
        chunk_data = {"word_count": 10, "token_count": 80}
        result = chunker.check_word_count_and_token_count(20, 30, chunk_data)
        assert result is False

    @patch("cognee.tasks.chunks.chunk_by_paragraph")
    def test_read_single_chunk(self, mock_chunk_by_paragraph, mock_document, mock_get_text):
        # Setup
        chunker = TextChunker(
            document=mock_document,
            get_text=mock_get_text,
            max_chunk_tokens=100,
            chunk_size=50
        )
        
        # Mock the chunk_by_paragraph function to return a single chunk
        chunk_data = {
            "text": "This is a test paragraph.",
            "word_count": 5,
            "token_count": 10,
            "chunk_id": uuid.uuid4(),
            "cut_type": "paragraph_end",
            "paragraph_ids": [1]
        }
        
        # Set up the mock to yield the chunk data when called
        mock_chunk_by_paragraph.return_value = iter([chunk_data])
        
        # Execute
        chunks = list(chunker.read())
        
        # Verify
        assert len(chunks) == 1
        assert isinstance(chunks[0], DocumentChunk)
        assert chunks[0].text == "This is a test paragraph."
        assert chunks[0].word_count == 5
        assert chunks[0].token_count == 10
        assert chunks[0].is_part_of == mock_document
        assert chunks[0].chunk_index == 0

    @patch("cognee.tasks.chunks.chunk_by_paragraph")
    def test_read_multiple_chunks(self, mock_chunk_by_paragraph, mock_document, mock_get_text):
        # Setup
        chunker = TextChunker(
            document=mock_document,
            get_text=mock_get_text,
            max_chunk_tokens=100,
            chunk_size=50
        )
        
        # Mock the chunk_by_paragraph function to return multiple chunks that will be combined
        chunk_data1 = {
            "text": "This is the first paragraph.",
            "word_count": 5,
            "token_count": 10,
            "chunk_id": uuid.uuid4(),
            "cut_type": "paragraph_end",
            "paragraph_ids": [1]
        }
        
        # Set up the mock to yield the chunk data when called
        mock_chunk_by_paragraph.return_value = iter([chunk_data1])
        
        # Execute
        chunks = list(chunker.read())
        
        # Verify
        assert len(chunks) == 1
        assert isinstance(chunks[0], DocumentChunk)
        assert chunks[0].text == "This is the first paragraph."
        assert chunks[0].word_count == 5
        assert chunks[0].token_count == 10
        assert chunks[0].is_part_of == mock_document
        assert chunks[0].chunk_index == 0

    @patch("cognee.tasks.chunks.chunk_by_paragraph")
    def test_read_exceeding_limits(self, mock_chunk_by_paragraph, mock_document, mock_get_text):
        # Setup
        chunker = TextChunker(
            document=mock_document,
            get_text=mock_get_text,
            max_chunk_tokens=15,  # Set a low limit to force splitting
            chunk_size=10
        )
        
        # Mock the chunk_by_paragraph function to return chunks that will exceed limits
        chunk_data1 = {
            "text": "This is the first paragraph.",
            "word_count": 5,
            "token_count": 10,
            "chunk_id": uuid.uuid4(),
            "cut_type": "paragraph_end",
            "paragraph_ids": [1]
        }
        chunk_data2 = {
            "text": "This is the second paragraph.",
            "word_count": 5,
            "token_count": 10,
            "chunk_id": uuid.uuid4(),
            "cut_type": "paragraph_end",
            "paragraph_ids": [2]
        }
        
        # Set up the mock to yield the chunk data when called
        # The second chunk should cause the limit to be exceeded
        mock_chunk_by_paragraph.return_value = iter([chunk_data1, chunk_data2])
        
        # Execute
        chunks = list(chunker.read())
        
        # Verify - we should get two chunks because the second one exceeds the limit
        assert len(chunks) == 2
        
        assert isinstance(chunks[0], DocumentChunk)
        assert chunks[0].text == "This is the first paragraph."
        assert chunks[0].word_count == 5
        assert chunks[0].token_count == 10
        assert chunks[0].is_part_of == mock_document
        assert chunks[0].chunk_index == 0
        
        assert isinstance(chunks[1], DocumentChunk)
        assert chunks[1].text == "This is the second paragraph."
        assert chunks[1].word_count == 5
        assert chunks[1].token_count == 10
        assert chunks[1].is_part_of == mock_document
        assert chunks[1].chunk_index == 1 
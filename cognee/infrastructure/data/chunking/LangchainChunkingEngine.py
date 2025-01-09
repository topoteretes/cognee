from __future__ import annotations
import re

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy


class LangchainChunkEngine:
    def __init__(self, chunk_strategy=None, source_data=None, chunk_size=None, chunk_overlap=None):
        self.chunk_strategy = chunk_strategy
        self.source_data = source_data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_data(
        self,
        chunk_strategy=None,
        source_data=None,
        chunk_size=None,
        chunk_overlap=None,
    ):
        """
        Chunk data based on the specified strategy.

        Parameters:
        - chunk_strategy: The strategy to use for chunking.
        - source_data: The data to be chunked.
        - chunk_size: The size of each chunk.
        - chunk_overlap: The overlap between chunks.

        Returns:
        - The chunked data.
        """

        if chunk_strategy == ChunkStrategy.CODE:
            chunked_data, chunk_number = self.chunk_data_by_code(
                source_data, self.chunk_size, self.chunk_overlap
            )

        elif chunk_strategy == ChunkStrategy.LANGCHAIN_CHARACTER:
            chunked_data, chunk_number = self.chunk_data_by_character(
                source_data, self.chunk_size, self.chunk_overlap
            )
        else:
            chunked_data, chunk_number = "Invalid chunk strategy.", [0, "Invalid chunk strategy."]
        return chunked_data, chunk_number

    def chunk_data_by_code(self, data_chunks, chunk_size, chunk_overlap=10, language=None):
        from langchain_text_splitters import (
            Language,
            RecursiveCharacterTextSplitter,
        )

        if language is None:
            language = Language.PYTHON
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        code_chunks = python_splitter.create_documents([data_chunks])

        only_content = [chunk.page_content for chunk in code_chunks]

        numbered_chunks = []
        for i, chunk in enumerate(code_chunks):
            numbered_chunk = [i + 1, chunk]
            numbered_chunks.append(numbered_chunk)

        return only_content, numbered_chunks

    def chunk_data_by_character(self, data_chunks, chunk_size=1500, chunk_overlap=10):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        data_chunks = splitter.create_documents([data_chunks])

        only_content = [chunk.page_content for chunk in data_chunks]

        numbered_chunks = []
        for i, chunk in enumerate(data_chunks):
            numbered_chunk = [i + 1, chunk]
            numbered_chunks.append(numbered_chunk)

        return only_content, numbered_chunks

from __future__ import annotations
import re

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy


class LangchainChunkEngine:
    """
    Handles chunking of data using specified strategies.
    """

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

        Select and apply a chunking strategy to the provided source data, returning the
        resulting chunks and their corresponding indices. If an invalid strategy is provided, an
        error message is returned instead.

        Parameters:
        -----------

            - chunk_strategy: The strategy to use for chunking; should be one of the predefined
              strategies. (default None)
            - source_data: The data to be chunked, passed into the chunking strategy. (default
              None)
            - chunk_size: The size of each chunk; determines how large each piece of data will
              be. (default None)
            - chunk_overlap: The amount of overlap between consecutive chunks; affects the
              continuity of data. (default None)

        Returns:
        --------

            A tuple containing the chunked data and its corresponding indices.
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
        """
        Chunk data specifically for code snippets.

        Utilize a text splitter to break down code into manageable chunks based on the provided
        size and overlap, returning the content and numbered indices of each chunk.

        Parameters:
        -----------

            - data_chunks: The code data that needs to be chunked.
            - chunk_size: The desired size of each code chunk.
            - chunk_overlap: The number of lines or characters that overlap between consecutive
              chunks. (default 10)
            - language: The programming language of the code, defaulting to Python if not
              specified. (default None)

        Returns:
        --------

            A tuple with the contents of the code chunks and their respective numbered lists.
        """
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
        """
        Chunk data based on character count.

        Apply a character-based text splitter to divide the input data into chunks of specified
        size and overlap, returning the content and the chunk indices.

        Parameters:
        -----------

            - data_chunks: The data to be chunked based on character count.
            - chunk_size: The maximum number of characters allowed in each chunk. (default 1500)
            - chunk_overlap: The number of characters that overlap between chunks. (default 10)

        Returns:
        --------

            A tuple comprising the content of the character chunks and their indexed
            representations.
        """
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

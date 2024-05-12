from __future__ import annotations
import re

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy



class LangchainChunkEngine():
    @staticmethod
    def chunk_data(
        chunk_strategy = None,
        source_data = None,
        chunk_size = None,
        chunk_overlap = None,
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
            chunked_data = LangchainChunkEngine.chunk_data_by_code(source_data,chunk_size, chunk_overlap)
        else:
            chunked_data = DefaultChunkEngine.chunk_data_by_paragraph(source_data,chunk_size, chunk_overlap)
        return chunked_data

    @staticmethod
    def chunk_data_by_code(data_chunks, chunk_size, chunk_overlap, language=None):
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

        return only_content


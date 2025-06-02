"""Chunking strategies for splitting text into smaller parts."""

from __future__ import annotations
import re
from cognee.shared.data_models import ChunkStrategy


# /Users/vasa/Projects/cognee/cognee/infrastructure/data/chunking/DefaultChunkEngine.py


class DefaultChunkEngine:
    """
    Manage the process of chunking data based on specified strategies.
    """

    def __init__(self, chunk_strategy=None, chunk_size=None, chunk_overlap=None):
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> list[str]:
        # Now that we have the separator, split the text
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = [_splits[0]] + splits
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

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
        -----------

            - chunk_strategy: The strategy to use for chunking the data. (default None)
            - source_data: The data to be chunked. (default None)
            - chunk_size: The size of each chunk. (default None)
            - chunk_overlap: The overlap between chunks. (default None)

        Returns:
        --------

            Returns the chunked data and the respective chunk numbers.
        """

        if self.chunk_strategy == ChunkStrategy.PARAGRAPH:
            chunked_data, chunk_number = self.chunk_data_by_paragraph(
                source_data, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        elif self.chunk_strategy == ChunkStrategy.SENTENCE:
            chunked_data, chunk_number = self.chunk_by_sentence(
                source_data, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        elif self.chunk_strategy == ChunkStrategy.EXACT:
            chunked_data, chunk_number = self.chunk_data_exact(
                source_data, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        else:
            chunked_data, chunk_number = "Invalid chunk strategy.", [0, "Invalid chunk strategy."]

        return chunked_data, chunk_number

    def chunk_data_exact(self, data_chunks, chunk_size, chunk_overlap):
        """
        Chunk data exactly by specified sizes and overlaps.

        Parameters:
        -----------

            - data_chunks: The chunks of data to be processed into exact sizes.
            - chunk_size: The defined size for each chunk to be created.
            - chunk_overlap: The number of overlapping characters between chunks.

        Returns:
        --------

            Returns the created chunks and their numbered indices.
        """
        data = "".join(data_chunks)
        chunks = []
        for i in range(0, len(data), chunk_size - chunk_overlap):
            chunks.append(data[i : i + chunk_size])
        numbered_chunks = []
        for i, chunk in enumerate(chunks):
            numbered_chunk = [i + 1, chunk]
            numbered_chunks.append(numbered_chunk)
        return chunks, numbered_chunks

    def chunk_by_sentence(self, data_chunks, chunk_size, chunk_overlap):
        """
        Chunk data into sentences based on specified sizes and overlaps.

        Parameters:
        -----------

            - data_chunks: The chunks of data to be processed into sentences.
            - chunk_size: The defined size for each chunk to be created.
            - chunk_overlap: The number of overlapping characters between chunks.

        Returns:
        --------

            Returns the resulting sentence chunks and their numbered indices.
        """
        # Split by periods, question marks, exclamation marks, and ellipses
        data = "".join(data_chunks)

        # The regular expression is used to find series of charaters that end with one the following chaacters (. ! ? ...)
        sentence_endings = r"(?<=[.!?…]) +"
        sentences = re.split(sentence_endings, data)

        sentence_chunks = []
        for sentence in sentences:
            if len(sentence) > chunk_size:
                chunks = self.chunk_data_exact(
                    data_chunks=[sentence], chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                sentence_chunks.extend(chunks)
            else:
                sentence_chunks.append(sentence)

        numbered_chunks = []
        for i, chunk in enumerate(sentence_chunks):
            numbered_chunk = [i + 1, chunk]
            numbered_chunks.append(numbered_chunk)
        return sentence_chunks, numbered_chunks

    def chunk_data_by_paragraph(self, data_chunks, chunk_size, chunk_overlap, bound=0.75):
        """
        Chunk data based on paragraphs while considering overlaps and boundaries.

        Parameters:
        -----------

            - data_chunks: The chunks of data to be processed into paragraphs.
            - chunk_size: The defined size for each chunk to be created.
            - chunk_overlap: The number of overlapping characters between chunks.
            - bound: A weighting factor to determine splitting within a chunk (default is 0.75).
              (default 0.75)

        Returns:
        --------

            Returns the paragraph chunks and their numbered indices.
        """
        data = "".join(data_chunks)
        total_length = len(data)
        chunks = []
        check_bound = int(bound * chunk_size)
        start_idx = 0
        chunk_splitter = "\n\n"

        if data.find("\n\n") == -1:
            chunk_splitter = "\n"

        while start_idx < total_length:
            # Set the end index to the minimum of start_idx + default_chunk_size or total_length
            end_idx = min(start_idx + chunk_size, total_length)

            # Find the next paragraph index within the current chunk and bound
            next_paragraph_index = data.find(chunk_splitter, start_idx + check_bound, end_idx)

            # If a next paragraph index is found within the current chunk
            if next_paragraph_index != -1:
                # Update end_idx to include the paragraph delimiter
                end_idx = next_paragraph_index + 2

            end_index = end_idx + chunk_overlap

            chunk_text = data[start_idx:end_index]

            while chunk_text[-1] != "." and end_index < total_length:
                chunk_text += data[end_index]
                end_index += 1

            end_idx = end_index - chunk_overlap

            chunks.append(chunk_text.replace("\n", "").strip())

            # Update start_idx to be the current end_idx
            start_idx = end_idx

        numbered_chunks = []
        for i, chunk in enumerate(chunks):
            numbered_chunk = [i + 1, chunk]
            numbered_chunks.append(numbered_chunk)

        return chunks, numbered_chunks

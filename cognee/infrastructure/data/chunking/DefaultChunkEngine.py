""" Chunking strategies for splitting text into smaller parts."""
from __future__ import annotations
import re
from cognee.shared.data_models import ChunkStrategy


class DefaultChunkEngine():
    @staticmethod
    def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
    ) -> list[str]:
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

        if chunk_strategy == ChunkStrategy.PARAGRAPH:
            chunked_data = DefaultChunkEngine.chunk_data_by_paragraph(source_data,chunk_size, chunk_overlap)
        elif chunk_strategy == ChunkStrategy.SENTENCE:
            chunked_data = DefaultChunkEngine.chunk_by_sentence(source_data, chunk_size, chunk_overlap)
        elif chunk_strategy == ChunkStrategy.EXACT:
            chunked_data = DefaultChunkEngine.chunk_data_exact(source_data, chunk_size, chunk_overlap)

        return chunked_data


    @staticmethod
    def chunk_data_exact(data_chunks, chunk_size, chunk_overlap):
        data = "".join(data_chunks)
        chunks = []
        for i in range(0, len(data), chunk_size - chunk_overlap):
            chunks.append(data[i:i + chunk_size])
        return chunks


    @staticmethod
    def chunk_by_sentence(data_chunks, chunk_size, overlap):
        # Split by periods, question marks, exclamation marks, and ellipses
        data = "".join(data_chunks)

        # The regular expression is used to find series of charaters that end with one the following chaacters (. ! ? ...)
        sentence_endings = r'(?<=[.!?…]) +'
        sentences = re.split(sentence_endings, data)

        sentence_chunks = []
        for sentence in sentences:
            if len(sentence) > chunk_size:
                chunks = DefaultChunkEngine.chunk_data_exact([sentence], chunk_size, overlap)
                sentence_chunks.extend(chunks)
            else:
                sentence_chunks.append(sentence)
        return sentence_chunks


    @staticmethod
    def chunk_data_by_paragraph(data_chunks, chunk_size, overlap, bound = 0.75):
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

            end_index = end_idx + overlap

            chunk_text = data[start_idx:end_index]

            while chunk_text[-1] != "." and end_index < total_length:
                chunk_text += data[end_index]
                end_index += 1

            end_idx = end_index - overlap

            chunks.append(chunk_text.replace("\n", "").strip())

            # Update start_idx to be the current end_idx
            start_idx = end_idx

        return chunks

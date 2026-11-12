from collections.abc import Iterator
from typing import Any
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine


def _get_pair_size(pair_text: str, tokenizer) -> int:
    """
    Calculate the size of a given text in terms of tokens.

    Counts the pair's tokens with the embedding engine's tokenizer, or counts it as
    three tokens when the engine has no tokenizer.

    Parameters:
    -----------

        - pair_text (str): The key:value pair text for which the token size is to be calculated.
        - tokenizer: The embedding engine's tokenizer, or None.

    Returns:
    --------

        - int: The number of tokens representing the text.
    """
    if tokenizer:
        return tokenizer.count_tokens(pair_text)
    return 3


def chunk_by_row(
    data: str,
    max_chunk_size,
) -> Iterator[dict[str, Any]]:
    """
    Chunk the input text by row while enabling exact text reconstruction.

    This function divides the given text data into smaller chunks on a line-by-line basis,
    ensuring that the size of each chunk is less than or equal to the specified maximum
    chunk size. It guarantees that when the generated chunks are concatenated, they
    reproduce the original text accurately. The tokenization process is handled by
    adapters compatible with the vector engine's embedding model.

    Parameters:
    -----------

        - data (str): The input text to be chunked.
        - max_chunk_size: The maximum allowed size for each chunk, in terms of tokens or
          words.
    """
    # Resolved once per call, not once per pair: the engine comes from the current
    # dataset's context, which cannot change mid-document (see chunk_by_sentence).
    tokenizer = get_embedding_engine().tokenizer
    current_chunk_list = []
    chunk_index = 0
    current_chunk_size = 0

    lines = data.split("\n\n")
    for line in lines:
        pairs_text = line.split(", ")

        for pair_text in pairs_text:
            pair_size = _get_pair_size(pair_text, tokenizer)
            if current_chunk_size > 0 and (current_chunk_size + pair_size > max_chunk_size):
                # Yield current cut chunk
                current_chunk = ", ".join(current_chunk_list)
                chunk_dict = {
                    "text": current_chunk,
                    "chunk_size": current_chunk_size,
                    "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
                    "chunk_index": chunk_index,
                    "cut_type": "row_cut",
                }

                yield chunk_dict

                # Start new chunk with current pair text
                current_chunk_list = []
                current_chunk_size = 0
                chunk_index += 1

            current_chunk_list.append(pair_text)
            current_chunk_size += pair_size

        # Yield row chunk
        current_chunk = ", ".join(current_chunk_list)
        if current_chunk:
            chunk_dict = {
                "text": current_chunk,
                "chunk_size": current_chunk_size,
                "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
                "chunk_index": chunk_index,
                "cut_type": "row_end",
            }

            yield chunk_dict

            chunk_index += 1

        # Start a fresh chunk for the next row so its pairs are not
        # accumulated onto this row's chunk and chunk indices stay
        # monotonically increasing. The reset must also run for a blank row
        # (consecutive separators produce ""), otherwise its empty pair leaks
        # into the next row's text, size, and chunk_id.
        current_chunk_list = []
        current_chunk_size = 0

from typing import Any, Dict, Iterator
from uuid import NAMESPACE_OID, uuid5

from .chunk_by_sentence import chunk_by_sentence


def chunk_by_paragraph(
    data: str,
    max_chunk_size,
    batch_paragraphs: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Chunk the input text by paragraph while enabling exact text reconstruction.

    This function divides the given text data into smaller chunks based on the specified
    maximum chunk size. It ensures that when the generated chunks are concatenated, they
    reproduce the original text accurately. The tokenization process is handled by adapters
    compatible with the vector engine's embedding model, and the function can operate in
    either batch mode or paragraph mode, based on the `batch_paragraphs` flag.

    Parameters:
    -----------

        - data (str): The input text to be chunked.
        - max_chunk_size: The maximum allowed size for each chunk, in terms of tokens or
          words.
        - batch_paragraphs (bool): Flag indicating whether to yield each paragraph as a
          separate chunk. If set to False, individual paragraphs are yielded as they are
          processed. (default True)
    """
    current_chunk = ""
    chunk_index = 0
    paragraph_ids = []
    last_cut_type = "default"
    current_chunk_size = 0

    for paragraph_id, sentence, sentence_size, end_type in chunk_by_sentence(
        data, maximum_size=max_chunk_size
    ):
        if current_chunk_size > 0 and (current_chunk_size + sentence_size > max_chunk_size):
            # Yield current chunk
            chunk_dict = {
                "text": current_chunk,
                "chunk_size": current_chunk_size,
                "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
                "paragraph_ids": paragraph_ids,
                "chunk_index": chunk_index,
                "cut_type": last_cut_type,
            }

            yield chunk_dict

            # Start new chunk with current sentence
            paragraph_ids = []
            current_chunk = ""
            current_chunk_size = 0
            chunk_index += 1

        paragraph_ids.append(paragraph_id)
        current_chunk += sentence
        current_chunk_size += sentence_size

        # Handle end of paragraph
        if end_type in ("paragraph_end", "sentence_cut") and not batch_paragraphs:
            # For non-batch mode, yield each paragraph separately
            chunk_dict = {
                "text": current_chunk,
                "chunk_size": current_chunk_size,
                "paragraph_ids": paragraph_ids,
                "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
                "chunk_index": chunk_index,
                "cut_type": end_type,
            }
            yield chunk_dict
            paragraph_ids = []
            current_chunk = ""
            current_chunk_size = 0
            chunk_index += 1

        if not end_type:
            end_type = "default"

        last_cut_type = end_type

    # Yield any remaining text
    if current_chunk:
        chunk_dict = {
            "text": current_chunk,
            "chunk_size": current_chunk_size,
            "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
            "paragraph_ids": paragraph_ids,
            "chunk_index": chunk_index,
            "cut_type": "sentence_cut" if last_cut_type == "word" else last_cut_type,
        }

        yield chunk_dict

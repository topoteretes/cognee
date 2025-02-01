from typing import Any, Dict, Iterator
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.databases.vector import get_vector_engine

from .chunk_by_sentence import chunk_by_sentence


def chunk_by_paragraph(
    data: str,
    max_chunk_tokens,
    paragraph_length: int = 1024,
    batch_paragraphs: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Chunks text by paragraph while preserving exact text reconstruction capability.
    When chunks are joined with empty string "", they reproduce the original text exactly.

    Notes:
        - Tokenization is handled using our tokenization adapters, ensuring compatibility with the vector engine's embedding model.
        - If `batch_paragraphs` is False, each paragraph will be yielded as a separate chunk.
        - Handles cases where paragraphs exceed the specified token or word limits by splitting them as needed.
        - Remaining text at the end of the input will be yielded as a final chunk.
    """
    current_chunk = ""
    current_word_count = 0
    chunk_index = 0
    paragraph_ids = []
    last_cut_type = None
    current_token_count = 0

    for paragraph_id, sentence, word_count, end_type in chunk_by_sentence(
        data, maximum_length=paragraph_length
    ):
        # Check if this sentence would exceed length limit
        embedding_engine = get_vector_engine().embedding_engine
        token_count = embedding_engine.tokenizer.count_tokens(sentence)

        if current_word_count > 0 and (
            current_word_count + word_count > paragraph_length
            or current_token_count + token_count > max_chunk_tokens
        ):
            # Yield current chunk
            chunk_dict = {
                "text": current_chunk,
                "word_count": current_word_count,
                "token_count": current_token_count,
                "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
                "paragraph_ids": paragraph_ids,
                "chunk_index": chunk_index,
                "cut_type": last_cut_type,
            }

            yield chunk_dict

            # Start new chunk with current sentence
            paragraph_ids = []
            current_chunk = ""
            current_word_count = 0
            current_token_count = 0
            chunk_index += 1

        paragraph_ids.append(paragraph_id)
        current_chunk += sentence
        current_word_count += word_count
        current_token_count += token_count

        # Handle end of paragraph
        if end_type in ("paragraph_end", "sentence_cut") and not batch_paragraphs:
            # For non-batch mode, yield each paragraph separately
            chunk_dict = {
                "text": current_chunk,
                "word_count": current_word_count,
                "token_count": current_token_count,
                "paragraph_ids": paragraph_ids,
                "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
                "chunk_index": chunk_index,
                "cut_type": end_type,
            }
            yield chunk_dict
            paragraph_ids = []
            current_chunk = ""
            current_word_count = 0
            current_token_count = 0
            chunk_index += 1

        last_cut_type = end_type

    # Yield any remaining text
    if current_chunk:
        chunk_dict = {
            "text": current_chunk,
            "word_count": current_word_count,
            "token_count": current_token_count,
            "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
            "paragraph_ids": paragraph_ids,
            "chunk_index": chunk_index,
            "cut_type": "sentence_cut" if last_cut_type == "word" else last_cut_type,
        }

        yield chunk_dict

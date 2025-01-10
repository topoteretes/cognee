from typing import Any, Dict, Iterator, Optional, Union
from uuid import NAMESPACE_OID, uuid5

import tiktoken

from cognee.infrastructure.databases.vector import get_vector_engine

from .chunk_by_sentence import chunk_by_sentence


def chunk_by_paragraph(
    data: str,
    max_tokens: Optional[Union[int, float]] = None,
    paragraph_length: int = 1024,
    batch_paragraphs: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Chunks text by paragraph while preserving exact text reconstruction capability.
    When chunks are joined with empty string "", they reproduce the original text exactly.
    """
    current_chunk = ""
    current_word_count = 0
    chunk_index = 0
    paragraph_ids = []
    last_cut_type = None
    current_token_count = 0
    if not max_tokens:
        max_tokens = float("inf")

    vector_engine = get_vector_engine()
    embedding_model = vector_engine.embedding_engine.model
    embedding_model = embedding_model.split("/")[-1]

    for paragraph_id, sentence, word_count, end_type in chunk_by_sentence(
        data, maximum_length=paragraph_length
    ):
        # Check if this sentence would exceed length limit

        tokenizer = tiktoken.encoding_for_model(embedding_model)
        token_count = len(tokenizer.encode(sentence))

        if current_word_count > 0 and (
            current_word_count + word_count > paragraph_length
            or current_token_count + token_count > max_tokens
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

from uuid import uuid5, NAMESPACE_OID
from typing import Dict, Any, Iterator
from .chunk_by_sentence import chunk_by_sentence

def chunk_by_paragraph(data: str, paragraph_length: int = 1024, batch_paragraphs: bool = True) -> Iterator[Dict[str, Any]]:
    """
    Chunks text by paragraph while preserving exact text reconstruction capability.
    When chunks are joined with empty string "", they reproduce the original text exactly.
    """
    current_chunk = ""
    current_word_count = 0
    chunk_index = 0
    last_paragraph_id = None
    last_cut_type = None
    
    for paragraph_id, _, sentence, word_count, end_type in chunk_by_sentence(data, maximum_length=paragraph_length):
        # Check if this sentence would exceed length limit
        if current_word_count > 0 and current_word_count + word_count > paragraph_length:
            # Yield current chunk
            chunk_dict = {
                "text": current_chunk,
                "word_count": current_word_count,
                "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
                "chunk_index": chunk_index,
                "cut_type": last_cut_type
            }
            
            if batch_paragraphs:
                chunk_dict["id"] = chunk_dict["chunk_id"]
            else:
                chunk_dict["id"] = last_paragraph_id
                
            yield chunk_dict
            
            # Start new chunk with current sentence
            current_chunk = sentence
            current_word_count = word_count
            chunk_index += 1
        else:
            # Just concatenate directly - no space handling
            current_chunk += sentence
            current_word_count += word_count
        
        # Handle end of paragraph
        if end_type in ("paragraph_end", "sentence_cut") and not batch_paragraphs:
            # For non-batch mode, yield each paragraph separately
            chunk_dict = {
                "text": current_chunk,
                "word_count": current_word_count,
                "id": paragraph_id,
                "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
                "chunk_index": chunk_index,
                "cut_type": end_type
            }
            yield chunk_dict
            current_chunk = ""
            current_word_count = 0
            chunk_index = 0
        
        last_cut_type = end_type
        last_paragraph_id = paragraph_id
    
    # Yield any remaining text
    if current_chunk:
        chunk_dict = {
            "text": current_chunk,
            "word_count": current_word_count,
            "chunk_id": uuid5(NAMESPACE_OID, current_chunk),
            "chunk_index": chunk_index,
            "cut_type": last_cut_type
        }
        
        if batch_paragraphs:
            chunk_dict["id"] = chunk_dict["chunk_id"]
        else:
            chunk_dict["id"] = last_paragraph_id
            
        yield chunk_dict
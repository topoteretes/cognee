from uuid import uuid4
from .chunk_by_sentence import chunk_by_sentence

def chunk_by_paragraph(data: str, paragraph_length: int = 1024):
    paragraph = ""
    paragraph_word_count = 0
    paragraph_chunk_index = 0

    for (paragraph_id, __, sentence, word_count, end_type) in chunk_by_sentence(data):
        if paragraph_word_count + word_count >= paragraph_length:
            yield dict(
                id = uuid4(),
                text = paragraph,
                word_count = paragraph_word_count,
                paragraph_id = paragraph_id,
                chunk_index = paragraph_chunk_index,
                is_end_chunk = False,
            )
            paragraph_chunk_index += 1
            paragraph_word_count = 0
            paragraph = ""

        paragraph += (" " if len(paragraph) > 0 else "") + sentence
        paragraph_word_count += word_count

        if end_type == "paragraph_end":
            yield dict(
                id = uuid4(),
                text = paragraph,
                word_count = paragraph_word_count,
                paragraph_id = paragraph_id,
                chunk_index = paragraph_chunk_index,
                is_end_chunk = True,
            )
            paragraph_chunk_index = 0
            paragraph_word_count = 0
            paragraph = ""

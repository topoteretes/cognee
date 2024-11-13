


from uuid import uuid4
from typing import Optional
from .chunk_by_word import chunk_by_word

def chunk_by_sentence(data: str, maximum_length: Optional[int] = None):
    sentence = ""
    paragraph_id = uuid4()
    chunk_index = 0
    word_count = 0

    for (word, word_type) in chunk_by_word(data):
        sentence += word
        word_count += 1

        if word_type == "paragraph_end" or word_type == "sentence_end" or ((maximum_length is not None) and (word_count == maximum_length)):
            yield (paragraph_id, chunk_index, sentence, word_count, word_type)
            sentence = ""
            word_count = 0
            paragraph_id = uuid4() if word_type == "paragraph_end" else paragraph_id
            chunk_index = 0 if word_type == "paragraph_end" else chunk_index + 1

    if len(sentence) > 0:
        yield (
            paragraph_id,
            chunk_index,
            sentence,
            word_count,
            "sentence_cut",
        )

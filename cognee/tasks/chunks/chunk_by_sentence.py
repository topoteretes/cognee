


from uuid import uuid4
from .chunk_by_word import chunk_by_word

def chunk_by_sentence(data: str):
    sentence = ""
    paragraph_id = uuid4()
    chunk_index = 0
    word_count = 0

    for (word, word_type) in chunk_by_word(data):
        sentence += (" " if len(sentence) > 0 else "") + word
        word_count += 1

        if word_type == "paragraph_end" or word_type == "sentence_end":
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

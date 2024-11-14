


from uuid import uuid4
from typing import Optional
from .chunk_by_word import chunk_by_word

def chunk_by_sentence(data: str, maximum_length: Optional[int] = None):
    sentence = ""
    paragraph_id = uuid4()
    chunk_index = 0
    word_count = 0
    section_end = False

    for (word, word_type) in chunk_by_word(data):
        sentence += word
        word_count += 1

        # this loop is to check if any letters come after a paragraph_end or sentence_end
        # and if that is not the case, preserve the word_type for the final yield in the
        # function
        if word_type in ["paragraph_end", "sentence_end"]:
            section_end = word_type
        else:
            for character in word:
                if character.isalpha():
                    section_end = "sentence_cut"
                    break

        if word_type in ["paragraph_end", "sentence_end"] or (maximum_length and (word_count == maximum_length)):
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
            section_end,
        )

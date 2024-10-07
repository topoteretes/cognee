from uuid import uuid5, NAMESPACE_OID
from .chunk_by_sentence import chunk_by_sentence

def chunk_by_paragraph(data: str, paragraph_length: int = 1024, batch_paragraphs = True):
    paragraph = ""
    last_cut_type = None
    last_paragraph_id = None
    paragraph_word_count = 0
    paragraph_chunk_index = 0

    for (paragraph_id, __, sentence, word_count, end_type) in chunk_by_sentence(data):
        if paragraph_word_count > 0 and paragraph_word_count + word_count > paragraph_length:
            if batch_paragraphs is True:
                chunk_id = uuid5(NAMESPACE_OID, paragraph)
                yield dict(
                    text = paragraph.strip(),
                    word_count = paragraph_word_count,
                    id = chunk_id, # When batching paragraphs, the paragraph_id is the same as chunk_id.
                                   # paragraph_id doens't mean anything since multiple paragraphs are merged.
                    chunk_id = chunk_id,
                    chunk_index = paragraph_chunk_index,
                    cut_type = last_cut_type,
                )
            else:
                yield dict(
                    text = paragraph.strip(),
                    word_count = paragraph_word_count,
                    id = last_paragraph_id,
                    chunk_id = uuid5(NAMESPACE_OID, paragraph),
                    chunk_index = paragraph_chunk_index,
                    cut_type = last_cut_type,
                )

            paragraph_chunk_index += 1
            paragraph_word_count = 0
            paragraph = ""

        paragraph += (" " if len(paragraph) > 0 else "") + sentence
        paragraph_word_count += word_count

        if end_type == "paragraph_end" or end_type == "sentence_cut":
            if batch_paragraphs is True:
                paragraph += "\n\n" if end_type == "paragraph_end" else ""
            else:
                yield dict(
                    text = paragraph.strip(),
                    word_count = paragraph_word_count,
                    paragraph_id = paragraph_id,
                    chunk_id = uuid5(NAMESPACE_OID, paragraph),
                    chunk_index = paragraph_chunk_index,
                    cut_type = end_type,
                )

                paragraph_chunk_index = 0
                paragraph_word_count = 0
                paragraph = ""

        last_cut_type = end_type
        last_paragraph_id = paragraph_id

    if len(paragraph) > 0:
        yield dict(
            chunk_id = uuid5(NAMESPACE_OID, paragraph),
            text = paragraph,
            word_count = paragraph_word_count,
            paragraph_id = last_paragraph_id,
            chunk_index = paragraph_chunk_index,
            cut_type = last_cut_type,
        )

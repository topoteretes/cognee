import re

SENTENCE_ENDINGS = r"[.;!?…]"
PARAGRAPH_ENDINGS = r"[\n\r]"

def chunk_by_word(data: str):
    last_processed_character = ""
    word = ""
    i = 0
    while i < len(data):
        character = data[i]

        if word == "" and (re.match(PARAGRAPH_ENDINGS, character) or character == " "):
            i = i + 1
            continue

        def is_real_paragraph_end():
            if re.match(SENTENCE_ENDINGS, last_processed_character):
                return True

            j = i + 1
            next_character = data[j] if j < len(data) else None
            while next_character is not None and (re.match(PARAGRAPH_ENDINGS, next_character) or next_character == " "):
                j += 1
                next_character = data[j] if j < len(data) else None
            if next_character and next_character.isupper():
                return True

            return False

        if re.match(PARAGRAPH_ENDINGS, character):
            yield (word, "paragraph_end" if is_real_paragraph_end() else "word")
            word = ""
            i = i + 1
            continue

        if character == " ":
            yield [word, "word"]
            word = ""
            i = i + 1
            continue

        word += character
        last_processed_character = character

        if re.match(SENTENCE_ENDINGS, character):
            # Check for ellipses.
            if i + 2 <= len(data) and data[i] == "." and data[i + 1] == "." and data[i + 2] == ".":
                word += ".."
                i = i + 2

            is_paragraph_end = i + 1 < len(data) and re.match(PARAGRAPH_ENDINGS, data[i + 1])
            yield (word, "paragraph_end" if is_paragraph_end else "sentence_end")
            word = ""

        i += 1

    if len(word) > 0:
        yield (word, "word")

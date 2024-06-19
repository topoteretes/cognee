import re

def chunk_by_word(data: str):
    sentence_endings = r"[.;!?…]"
    paragraph_endings = r"[\n\r]"

    word = ""
    i = 0

    while i < len(data):
        character = data[i]

        if word == "" and (re.match(paragraph_endings, character) or character == " "):
            i = i + 1
            continue

        if re.match(paragraph_endings, character):
            yield (word, "paragraph_end")
            word = ""
            i = i + 1
            continue

        if character == " ":
            yield [word, "word"]
            word = ""
            i = i + 1
            continue

        word += character

        if re.match(sentence_endings, character):
            # Check for ellipses.
            if i + 2 <= len(data) and data[i] == "." and data[i + 1] == "." and data[i + 2] == ".":
                word += ".."
                i = i + 2

            is_paragraph_end = i + 1 < len(data) and re.match(paragraph_endings, data[i + 1])
            yield (word, "paragraph_end" if is_paragraph_end else "sentence_end")
            word = ""

        i += 1

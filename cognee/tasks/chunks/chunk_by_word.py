import re

SENTENCE_ENDINGS = r"[.;!?…]"
PARAGRAPH_ENDINGS = r"[\n\r]"

def is_real_paragraph_end(last_processed_character, i, data):
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

def chunk_by_word(data: str):
    """
    Chunks text into words and endings while preserving whitespace.
    Whitespace is included with the preceding word.
    Outputs can be joined with "" to recreate the original input.
    """
    last_processed_character = ""
    current_chunk = ""
    i = 0
    
    # Handle leading whitespace if any
    while i < len(data) and (re.match(PARAGRAPH_ENDINGS, data[i]) or data[i] == " "):
        current_chunk += data[i]
        i += 1
    if current_chunk:
        yield (current_chunk, "word")
        current_chunk = ""
    
    while i < len(data):
        character = data[i]
            
        if re.match(PARAGRAPH_ENDINGS, character):
            if current_chunk:
                yield (current_chunk, "word")
                current_chunk = ""
            yield (character, "paragraph_end" if is_real_paragraph_end(last_processed_character, i, data) else "word")
            i += 1
            continue
            
        current_chunk += character
        last_processed_character = character
        
        if character == " ":
            yield (current_chunk, "word")
            current_chunk = ""
            i += 1
            continue
        
        if re.match(SENTENCE_ENDINGS, character):
            # Check for ellipses
            if i + 2 < len(data) and data[i:i+3] == "...":
                current_chunk += ".."
                i += 2
                
            # Look ahead for whitespace
            next_i = i + 1
            while next_i < len(data) and data[next_i] == " ":
                current_chunk += data[next_i]
                next_i += 1
                
            is_paragraph_end = next_i < len(data) and re.match(PARAGRAPH_ENDINGS, data[next_i])
            yield (current_chunk, "paragraph_end" if is_paragraph_end else "sentence_end")
            current_chunk = ""
            i = next_i
            continue
            
        i += 1
        
    if current_chunk:
        yield (current_chunk, "word")
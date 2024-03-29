def is_text_content(content):
    # Check for null bytes
    if b'\0' in content:
        return False

    # Check for common text encodings (BOMs)
    if content.startswith((b'\xEF\xBB\xBF',  # UTF-8
                            b'\xFF\xFE',      # UTF-16 LE
                            b'\xFE\xFF',      # UTF-16 BE
                            b'\x00\x00\xFE\xFF',  # UTF-32 LE
                            b'\xFF\xFE\x00\x00',  # UTF-32 BE
                            )):
        return True

    # Check for ASCII characters
    if all(0x20 <= byte <= 0x7E or byte in (b'\n', b'\r', b'\t') for byte in content):
        return True

    # Check for common line break characters
    if b'\n' in content or b'\r' in content:
        return True

    # If no obvious indicators found, assume it's a text file
    return True

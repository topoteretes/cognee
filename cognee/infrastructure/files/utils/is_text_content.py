def is_text_content(content):
    """Check if the content is text."""
    # Check for null bytes
    if b"\0" in content:
        return False

    # Check for common text encodings (BOMs)
    if content.startswith(
        (
            b"\xef\xbb\xbf",  # UTF-8
            b"\xff\xfe",  # UTF-16 LE
            b"\xfe\xff",  # UTF-16 BE
            b"\x00\x00\xfe\xff",  # UTF-32 LE
            b"\xff\xfe\x00\x00",  # UTF-32 BE
        )
    ):
        return True

    # Check for ASCII characters
    if all(0x20 <= byte <= 0x7E or byte in (b"\n", b"\r", b"\t") for byte in content):
        return True

    # Check for common line break characters
    if b"\n" in content or b"\r" in content:
        return True

    # If no obvious indicators found, assume it's a text file
    return True

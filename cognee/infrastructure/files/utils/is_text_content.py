from cognee.infrastructure.files.utils.decode_text_sample import TEXT_BOMS


def is_text_content(content):
    """
    Determine if the content is text-based.

    This function checks for various indicators of text content, including byte order marks
    (BOMs), null bytes, ASCII character ranges, and common line breaks. If none of these
    indicators are present, the function defaults to assuming the content is text-based.

    Parameters:
    -----------

        - content: The byte content to be checked for text characteristics.

    Returns:
    --------

        - bool: Returns True if the content is determined to be text, False otherwise.
    """
    # Check for common text encodings (BOMs) FIRST. A UTF-16/UTF-32 text file is
    # full of null bytes, so the null-byte check below would reject it as binary
    # before its BOM was ever read, and its content (PII included) would never be
    # scanned. The marks live with the decoder they pair with so the two agree.
    if content.startswith(TEXT_BOMS):
        return True

    # Check for null bytes
    if b"\0" in content:
        return False

    # Check for ASCII characters
    if all(0x20 <= byte <= 0x7E or byte in (b"\n", b"\r", b"\t") for byte in content):
        return True

    # Check for common line break characters
    if b"\n" in content or b"\r" in content:
        return True

    # If no obvious indicators found, assume it's a text file
    return True

import codecs


def is_text_content(content):
    """
    Determine if the content is text-based.

    This function checks for various indicators of text content, including null bytes, byte
    order marks (BOMs), ASCII character ranges, and common line breaks. If none of these
    indicators are present, the function defaults to assuming the content is text-based.

    Parameters:
    -----------

        - content: The byte content to be checked for text characteristics.

    Returns:
    --------

        - bool: Returns True if the content is determined to be text, False otherwise.
    """
    # Recognized text BOMs take precedence over null bytes in UTF-16/UTF-32.
    if content.startswith(
        (
            codecs.BOM_UTF8,
            codecs.BOM_UTF16_LE,
            codecs.BOM_UTF16_BE,
            codecs.BOM_UTF32_LE,
            codecs.BOM_UTF32_BE,
        )
    ):
        return True

    # Null-containing content without a recognized BOM is binary.
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

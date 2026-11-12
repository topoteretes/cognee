def is_text_content(content: bytes) -> bool:
    """
    Determine if the content is text-based.

    This function checks for various indicators of text content, including byte
    order marks (BOMs), UTF-16 patterns, null bytes, ASCII character ranges,
    and common line breaks. If none of these indicators are present, the function
    defaults to assuming the content is text-based.

    Parameters:
    -----------

        - content: The byte content to be checked for text characteristics.

    Returns:
    --------

        - bool: Returns True if the content is determined to be text, False otherwise.
    """
    if not content:
        return True

    # Check for common text encodings (BOMs) FIRST, before checking for null bytes,
    # because UTF-16 and UTF-32 contain valid null bytes for standard characters.
    # Note: 4-byte BOMs must be tested before 2-byte BOMs.
    if content.startswith(
        (
            b"\x00\x00\xfe\xff",  # UTF-32 BE
            b"\xff\xfe\x00\x00",  # UTF-32 LE
            b"\xef\xbb\xbf",  # UTF-8
            b"\xff\xfe",  # UTF-16 LE
            b"\xfe\xff",  # UTF-16 BE
        )
    ):
        return True

    # If null bytes are present, check if it is UTF-16 (LE or BE) without a BOM
    if b"\x00" in content:
        if len(content) >= 4:
            even_bytes = content[: len(content) - (len(content) % 2)]
            zero_odds = even_bytes[1::2].count(0)
            zero_evens = even_bytes[0::2].count(0)
            # In ASCII/Latin text encoded in UTF-16, null bytes are concentrated in either odd or even positions
            if (zero_odds > 0 or zero_evens > 0) and zero_odds != zero_evens:
                encodings = (
                    ("utf-16-le", "utf-16-be")
                    if zero_odds > zero_evens
                    else ("utf-16-be", "utf-16-le")
                )
                for enc in encodings:
                    try:
                        decoded = even_bytes.decode(enc)
                        ascii_printable = sum(
                            1 for ch in decoded if 0x20 <= ord(ch) <= 0x7E or ch in "\n\r\t"
                        )
                        if len(decoded) > 0 and (ascii_printable / len(decoded)) >= 0.70:
                            return True
                    except (UnicodeDecodeError, ValueError):
                        pass
        return False

    # Check for ASCII characters
    if all(0x20 <= byte <= 0x7E or byte in (b"\n", b"\r", b"\t") for byte in content):
        return True

    # Check for common line break characters
    if b"\n" in content or b"\r" in content:
        return True

    # If no obvious indicators found, assume it's a text file
    return True


def decode_text_content(content: bytes) -> str:
    """
    Decode raw byte content of a text file into a string, properly handling
    UTF-8, UTF-8 BOM, UTF-16 LE/BE (with or without BOM), and UTF-32.
    """
    if not content:
        return ""

    # Check for BOMs
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig", errors="replace")
    if content.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return content.decode("utf-32", errors="replace")
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16", errors="replace")

    # If null bytes are present, inspect for UTF-16 without BOM
    if b"\x00" in content and len(content) >= 4:
        even_bytes = content[: len(content) - (len(content) % 2)]
        zero_odds = even_bytes[1::2].count(0)
        zero_evens = even_bytes[0::2].count(0)
        encodings = (
            ("utf-16-le", "utf-16-be") if zero_odds >= zero_evens else ("utf-16-be", "utf-16-le")
        )
        for enc in encodings:
            try:
                decoded = even_bytes.decode(enc)
                ascii_printable = sum(
                    1 for ch in decoded if 0x20 <= ord(ch) <= 0x7E or ch in "\n\r\t"
                )
                if len(decoded) > 0 and (ascii_printable / len(decoded)) >= 0.70:
                    return decoded
            except (UnicodeDecodeError, ValueError):
                pass

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass

    return content.decode("utf-8", errors="replace")

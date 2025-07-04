import hashlib


def get_text_content_hash(text: str) -> str:
    encoded_text = text.encode("utf-8")
    return hashlib.md5(encoded_text).hexdigest()

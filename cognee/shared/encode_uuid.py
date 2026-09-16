from uuid import UUID


def encode_uuid(uuid: UUID | str) -> str:
    """Encode a UUID instance or string into a 36-character base-52 encoded representation."""
    if isinstance(uuid, str):
        uuid = UUID(uuid)
    elif not isinstance(uuid, UUID):
        raise TypeError(f"Expected UUID or str, got {type(uuid).__name__}")

    uuid_int = uuid.int
    base = 52
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    encoded = ""
    while len(encoded) < 36:
        uuid_int, remainder = divmod(uuid_int, base)
        uuid_int = uuid_int * 8
        encoded = charset[remainder] + encoded

    return encoded


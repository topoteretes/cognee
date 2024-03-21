from uuid import UUID

def encode_uuid(uuid: UUID) -> str:
    uuid_int = uuid.int
    base = 52
    charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    encoded = ''
    while len(encoded) < 36:
        uuid_int, remainder = divmod(uuid_int, base)
        uuid_int = uuid_int * 8
        encoded = charset[remainder] + encoded

    return encoded

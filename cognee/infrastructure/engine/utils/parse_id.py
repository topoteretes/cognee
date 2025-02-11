from uuid import UUID


def parse_id(id: any):
    if isinstance(id, str):
        try:
            return UUID(id)
        except Exception:
            pass
    return id

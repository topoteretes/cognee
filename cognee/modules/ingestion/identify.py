from uuid import uuid5, NAMESPACE_OID
from .data_types import IngestionData

def identify(data: IngestionData) -> str:
    data_id: str = data.get_identifier()

    return uuid5(NAMESPACE_OID, data_id)

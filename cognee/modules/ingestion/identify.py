from uuid import uuid5, UUID
from .data_types import IngestionData

null_uuid: UUID = UUID("00000000-0000-0000-0000-000000000000")

def identify(data: IngestionData) -> str:
    data_id: str = data.get_identifier()

    return str(uuid5(null_uuid, data_id)).replace("-", "")

from uuid import UUID
from .data_types import IngestionData

from cognee.modules.users.models import User
from cognee.modules.data.methods import get_unique_data_id


async def identify(data: IngestionData, user: User) -> UUID:
    data_content_hash: str = data.get_identifier()

    return await get_unique_data_id(data_identifier=data_content_hash, user=user)

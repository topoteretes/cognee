from uuid import uuid5, NAMESPACE_OID
from .data_types import IngestionData

from cognee.modules.users.models import User


def identify(data: IngestionData, user: User) -> str:
    data_content_hash: str = data.get_identifier()

    # return UUID hash of file contents + owner id
    return uuid5(NAMESPACE_OID, f"{data_content_hash}{user.id}")

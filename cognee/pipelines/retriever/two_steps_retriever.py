from uuid import UUID
from enum import Enum
from typing import Callable, Dict
from cognee.shared.utils import send_telemetry
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.permissions.methods import get_document_ids_for_user

async def two_step_retriever(query: Dict[str, str], user: User = None) -> list:
    if user is None:
        user = await get_default_user()

    if user is None:
        raise PermissionError("No user found in the system. Please create a user.")

    own_document_ids = await get_document_ids_for_user(user.id)
    retrieved_results = await run_two_step_retriever(query, user)

    filtered_search_results = []


    return retrieved_results


async def run_two_step_retriever(query: str, user, community_filter = []) -> list:
    raise(NotImplementedError)
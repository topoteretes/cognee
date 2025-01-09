from cognee.modules.users.get_fastapi_users import get_fastapi_users
from cognee.modules.users.models.User import UserRead


def get_verify_router():
    return get_fastapi_users().get_verify_router(UserRead)

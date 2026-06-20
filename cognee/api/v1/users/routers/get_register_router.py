from cognee.modules.users.get_fastapi_users import get_fastapi_users
from cognee.modules.users.models.User import UserRead, UserCreate


def get_register_router():
    return get_fastapi_users().get_register_router(UserRead, UserCreate)

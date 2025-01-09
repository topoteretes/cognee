from cognee.modules.users.get_fastapi_users import get_fastapi_users


def get_reset_password_router():
    return get_fastapi_users().get_reset_password_router()

from cognee.modules.users.get_fastapi_users import get_fastapi_users
from cognee.modules.users.authentication.get_client_auth_backend import get_client_auth_backend


def get_auth_router():
    auth_backend = get_client_auth_backend()
    return get_fastapi_users().get_auth_router(auth_backend)

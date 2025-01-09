from ..get_fastapi_users import get_fastapi_users

fastapi_users = get_fastapi_users()

get_authenticated_user = fastapi_users.current_user(active=True, verified=True)

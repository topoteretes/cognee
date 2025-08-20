from ..get_fastapi_users import get_fastapi_users

# Create optional authenticated user dependency using FastAPI Users' built-in optional parameter
fastapi_users = get_fastapi_users()
get_optional_authenticated_user = fastapi_users.current_user(
    optional=True,  # Returns None instead of raising HTTPException(401)
    active=True     # Still require users to be active when authenticated
)

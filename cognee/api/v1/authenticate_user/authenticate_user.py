from cognee.infrastructure.databases.relational.user_authentication.users import authenticate_user_method


async def authenticate_user():
    """
    This function is used to authenticate a user.
    """
    output = await authenticate_user_method()
    return output

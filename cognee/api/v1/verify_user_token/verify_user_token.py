from cognee.infrastructure.databases.relational.user_authentication.authentication_db import user_check_token



async def verify_user_token(token: str):

    output = await user_check_token(token=token)
    return output
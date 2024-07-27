from cognee.infrastructure.databases.relational.user_authentication.users import user_check_token



async def verify_user_token(token: str):

    output = await user_check_token(token=token)
    return output
from cognee.infrastructure.databases.relational.user_authentication.users import (
    authenticate_user_method,
)


async def authenticate_user(email: str, password: str):
    """
    This function is used to authenticate a user.
    """
    output = await authenticate_user_method(email=email, password=password)
    return output


if __name__ == "__main__":
    import asyncio

    # Define an example user
    example_email = "example@example.com"
    example_password = "securepassword123"
    example_is_superuser = False

    # Create an event loop and run the create_user function
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(authenticate_user(example_email, example_password))

    # Print the result
    print(result)

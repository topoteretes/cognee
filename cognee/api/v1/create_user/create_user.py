from cognee.infrastructure.databases.relational.user_authentication.users import create_user_method



async def create_user(email: str, password: str, is_superuser: bool = False):
    output = await create_user_method(email=email, password=password, is_superuser=is_superuser)
    return output


if __name__ == "__main__":
    import asyncio
    # Define an example user
    example_email = "example@example.com"
    example_password = "securepassword123"
    example_is_superuser = False

    # Create an event loop and run the create_user function
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(create_user(example_email, example_password, example_is_superuser))

    # Print the result
    print(result)
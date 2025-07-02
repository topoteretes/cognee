import os
import re
import json
import uuid
from typing import Optional
from fastapi import Depends, Request, Response
from fastapi_users.exceptions import UserNotExists
from fastapi_users import BaseUserManager, UUIDIDMixin
from fastapi_users.db import SQLAlchemyUserDatabase
from contextlib import asynccontextmanager

from .models import User
from .get_user_db import get_user_db
from .methods.get_user_by_email import get_user_by_email


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = os.getenv(
        "FASTAPI_USERS_RESET_PASSWORD_TOKEN_SECRET", "super_secret"
    )
    verification_token_secret = os.getenv("FASTAPI_USERS_VERIFICATION_TOKEN_SECRET", "super_secret")

    # async def get(self, id: models.ID) -> models.UP:
    #     """
    #     Get a user by id.

    #     :param id: Id. of the user to retrieve.
    #     :raises UserNotExists: The user does not exist.
    #     :return: A user.
    #     """
    #     user = await get_user(id)

    #     if user is None:
    #         raise UserNotExists()

    #     return user

    async def get_by_email(self, user_email: str) -> Optional[User]:
        user = await get_user_by_email(user_email)

        if user is None:
            raise UserNotExists()

        return user

    async def on_after_login(
        self, user: User, request: Optional[Request] = None, response: Optional[Response] = None
    ):
        access_token_cookie = response.headers.get("Set-Cookie")
        match = re.search(
            r"(?i)\bSet-Cookie:\s*([^=]+)=([^;]+)", f"Set-Cookie: {access_token_cookie}"
        )
        if match:
            access_token = match.group(2)
            response.status_code = 200
            response.body = json.dumps(
                {"access_token": access_token, "token_type": "bearer"}
            ).encode(encoding="utf-8")
            response.headers.append("Content-Type", "application/json")

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"Verification requested for user {user.id}. Verification token: {token}")


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


get_user_manager_context = asynccontextmanager(get_user_manager)

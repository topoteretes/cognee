import uuid
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, UUIDIDMixin
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.exceptions import UserNotExists


from .models.User import User
from .get_user_db import get_user_db
from .methods.get_user_by_email import get_user_by_email


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    async def get_by_email(self, user_email: str) -> Optional[User]:
        user = await get_user_by_email(user_email)

        if user is None:
            raise UserNotExists()

        return user

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

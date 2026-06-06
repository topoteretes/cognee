import os
import re
import json
import uuid
import logging
from sqlalchemy import select
from typing import Optional
from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users import BaseUserManager, UUIDIDMixin, exceptions
from fastapi_users.db import SQLAlchemyUserDatabase
from pwdlib.exceptions import UnknownHashError
from contextlib import asynccontextmanager

from .models import User
from .get_user_db import get_user_db
from cognee.modules.users.models.UserApiKey import UserApiKey
from cognee.modules.users.api_key.hash_api_key import prepare_api_key
from cognee.infrastructure.databases.relational import get_relational_engine

logger = logging.getLogger(__name__)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = os.getenv(
        "FASTAPI_USERS_RESET_PASSWORD_TOKEN_SECRET", "super_secret"
    )
    verification_token_secret = os.getenv("FASTAPI_USERS_VERIFICATION_TOKEN_SECRET", "super_secret")

    async def on_after_login(
        self, user: User, request: Optional[Request] = None, response: Optional[Response] = None
    ):
        logger.info("User %s has logged in.", user.id)

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        logger.info("User %s has registered.", user.id)

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info("User %s has forgot their password. Reset token: %s", user.id, token)

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info("Verification requested for user %s. Verification token: %s", user.id, token)

    async def authenticate(self, credentials: OAuth2PasswordRequestForm) -> Optional[User]:
        try:
            user = await self.get_by_email(credentials.username)
        except exceptions.UserNotExists:
            self.password_helper.hash(credentials.password)
            return None

        try:
            verified, updated_password_hash = self.password_helper.verify_and_update(
                credentials.password, user.hashed_password
            )
        except UnknownHashError:
            raise HTTPException(
                status_code=400,
                detail="This user does not have a password. Use API key authentication.",
            )

        if not verified:
            return None

        if updated_password_hash is not None:
            await self.user_db.update(user, {"hashed_password": updated_password_hash})

        return user

    async def get_by_token(self, token: str) -> Optional[User]:
        relational_engine = get_relational_engine()
        prepared_api_key = prepare_api_key(token)

        async with relational_engine.get_async_session() as session:
            user_api_key = (
                await session.execute(select(UserApiKey).filter_by(api_key=prepared_api_key))
            ).scalar()

            if user_api_key:
                user = await self.get(user_api_key.user_id)

                return user


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


get_user_manager_context = asynccontextmanager(get_user_manager)

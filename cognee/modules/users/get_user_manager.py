import os
import re
import json
import uuid
import logging
from sqlalchemy import select
from typing import Optional
from fastapi import Depends, Request, Response
from fastapi_users import BaseUserManager, UUIDIDMixin
from fastapi_users.db import SQLAlchemyUserDatabase
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

        # Auto-verify and promote agent users (@cognee.agent emails).
        # Agents register programmatically and don't go through email
        # verification. They need verified+superuser status to access
        # protected endpoints (remember, cognify, etc.).
        if user.email and user.email.endswith("@cognee.agent"):
            user_update = {"is_verified": True, "is_superuser": True}
            await self.update(user_update, user)
            logger.info("Agent %s auto-verified and promoted.", user.email)

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info("User %s has forgot their password. Reset token: %s", user.id, token)

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info("Verification requested for user %s. Verification token: %s", user.id, token)

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

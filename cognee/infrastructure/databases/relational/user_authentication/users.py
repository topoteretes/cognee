import hashlib
import uuid
from typing import Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, models
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.exceptions import UserAlreadyExists
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from cognee.infrastructure.databases.relational.user_authentication.authentication_db import User, get_user_db, \
    get_async_session
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users.authentication import JWTStrategy
from cognee.infrastructure.databases.relational.user_authentication.schemas import UserRead, UserCreate
from contextlib import asynccontextmanager

SECRET = "SECRET"


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

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


bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)


async def get_user_permissions(user: User, session: Session):
    permissions = set()
    for group in user.groups:
        permissions.update(permission.name for permission in group.permissions)
    return permissions

def has_permission(permission: str):
    async def permission_checker(user: User = Depends(current_active_user), session: Session = Depends(get_user_db)):
        user_permissions = await get_user_permissions(user, session)
        if permission not in user_permissions:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
        return True
    return Depends(permission_checker)


async def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Define context managers for dependencies
get_async_session_context = asynccontextmanager(get_async_session)
get_user_db_context = asynccontextmanager(get_user_db)
get_user_manager_context = asynccontextmanager(get_user_manager)

async def create_user_method(email: str, password: str, is_superuser: bool = False):
    try:
        async with get_async_session_context() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    user = await user_manager.create(
                        UserCreate(email=email, password=password, is_superuser=is_superuser)
                    )
                    print(f"User created: {user.email}")
    except UserAlreadyExists:
        print(f"User {email} already exists")

async def authenticate_user_method(email: str, password: str) -> Optional[User]:
    try:
        async with get_async_session_context() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    credentials = OAuth2PasswordRequestForm(username=email, password=password)
                    user = await user_manager.authenticate(credentials)
                    if user is None or not user.is_active:
                        return None
                    return user
    except Exception as e:
        print(f"Error during authentication: {e}")
        return None

async def reset_user_password_method(email: str, new_password: str) -> bool:
    async with get_async_session_context() as session:
        async with get_user_db_context(session) as user_db:
            user = await user_db.get_by_email(email)
            if not user:
                return False
            user.hashed_password = await hash_password(new_password)
            await user_db.update(user)
            return True

# async def generate_verification_token(email: str, tokens_db: dict) -> str:
#     async with get_async_session_context() as session:
#         async with get_user_db_context(session) as user_db:
#             if not await user_db.get_by_email(email):
#                 raise ValueError("User does not exist")
#             token = str(uuid.uuid4())
#             tokens_db[token] = email
#             return token

# async def verify_user_method(token: str, tokens_db: dict) -> bool:
#     async with get_async_session_context() as session:
#         async with get_user_db_context(session) as user_db:
#             email = tokens_db.get(token)
#             if not email or not await user_db.get_by_email(email):
#                 return False
#             user = await user_db.get_by_email(email)
#             user.is_verified = True
#             await user_db.update(user)
#             return True


async def user_create_token(user: User) -> Optional[str]:
    try:
        async with get_async_session_context() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    if user is None:
                        return None
                    strategy = get_jwt_strategy()
                    token = await strategy.write_token(user)
                    if token is not None:
                        return token
                    else:
                        return None
    except:
        return None

async def user_check_token(token: str) -> bool:
    try:
        async with get_async_session_context() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    if token is None:
                        return False
                    strategy = get_jwt_strategy()
                    user = await strategy.read_token(token, user_manager)
                    if user is None or not user.is_active:
                        return False
                    else:
                        return True
    except:
        return False


import hashlib
import uuid
from typing import AsyncGenerator, Generator, Optional

from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy import create_engine, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from cognee.infrastructure.databases.relational import get_relationaldb_config, Base
from sqlalchemy import Column, String, ForeignKey, Table, Integer
from contextlib import asynccontextmanager
from fastapi_users.exceptions import UserAlreadyExists
from fastapi_users.authentication import JWTStrategy
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session, relationship
from cognee.infrastructure.databases.relational.user_authentication.schemas import UserRead, UserCreate
from cognee.infrastructure.databases.relational.user_authentication.users import get_user_manager, get_jwt_strategy
from fastapi.security import OAuth2PasswordRequestForm
# Association table for many-to-many relationship between users and groups
user_group = Table('user_group', Base.metadata,
                   Column('user_id', UUID, ForeignKey('users.id')),
                   Column('group_id', UUID, ForeignKey('groups.id')))

# Association table for many-to-many relationship between groups and permissions
group_permission = Table('group_permission', Base.metadata,
                         Column('group_id', UUID, ForeignKey('groups.id')),
                         Column('permission_id', UUID, ForeignKey('permissions.id')))



class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = 'users'
    groups = relationship('Group', secondary=user_group, back_populates='users')

class Group(Base):
    __tablename__ = 'groups'
    id = Column(UUID, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    users = relationship('users', secondary=user_group, back_populates='groups')
    permissions = relationship('Permission', secondary=group_permission, back_populates='groups')

class Permission(Base):
    __tablename__ = 'permissions'
    id = Column(UUID, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    groups = relationship('Group', secondary=group_permission, back_populates='permissions')

class ACL(Base):
    __tablename__ = 'acls'
    id = Column(UUID, primary_key=True, index=True)
    document_id = Column(UUID, ForeignKey('documents.id'))
    user_id = Column(UUID, ForeignKey('users.id'), nullable=True)
    group_id = Column(UUID, ForeignKey('groups.id'), nullable=True)
    permission = Column(String)  # 'read', 'write', 'execute'
    document = relationship('dataset_data', back_populates='acls')
    user = relationship('users', back_populates='acls')
    group = relationship('groups', back_populates='acls')


relational_config = get_relationaldb_config()



engine = relational_config.create_engine()
async_session_maker = async_sessionmaker(engine.engine, expire_on_commit=False)

async def create_db_and_tables():
    async with engine.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
    # yield async_session_maker

async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


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
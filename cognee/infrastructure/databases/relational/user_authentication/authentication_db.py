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
# from cognee.infrastructure.databases.relational.user_authentication.users import get_user_manager, get_jwt_strategy
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
    acls = relationship('ACL', back_populates='user')

class Group(Base):
    __tablename__ = 'groups'
    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    name = Column(String, unique=True, index=True)
    users = relationship('User', secondary=user_group, back_populates='groups')
    permissions = relationship('Permission', secondary=group_permission, back_populates='groups')
    acls = relationship('ACL', back_populates='group')
class Permission(Base):
    __tablename__ = 'permissions'
    id = Column(UUID, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    groups = relationship('Group', secondary=group_permission, back_populates='permissions')

class ACL(Base):
    __tablename__ = 'acls'
    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    document_id = Column(UUID, ForeignKey('dataset_data.id'))
    user_id = Column(UUID, ForeignKey('users.id'), nullable=True)
    group_id = Column(UUID, ForeignKey('groups.id'), nullable=True)
    permission = Column(String)  # 'read', 'write', 'execute'
    document = relationship('DatasetData', back_populates='acls')
    user = relationship('User', back_populates='acls')
    group = relationship('Group', back_populates='acls')

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



from typing import AsyncGenerator, Generator

from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy import create_engine, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from cognee.infrastructure.databases.relational import get_relationaldb_config, Base
from sqlalchemy import Column, String, ForeignKey, Table, Integer


from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session, relationship

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
    users = relationship('User', secondary=user_group, back_populates='groups')
    permissions = relationship('Permission', secondary=group_permission, back_populates='groups')

class Permission(Base):
    __tablename__ = 'permissions'
    id = Column(UUID, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    groups = relationship('Group', secondary=group_permission, back_populates='permissions')


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

from uuid import UUID as uuid_UUID
from sqlalchemy import ForeignKey, UUID, Column
from sqlalchemy.orm import relationship, Mapped
from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from .Principal import Principal
from .UserGroup import UserGroup

class User(SQLAlchemyBaseUserTableUUID, Principal):
    __tablename__ = "users"

    id = Column(UUID(as_uuid = True), ForeignKey("principals.id"), primary_key = True)

    groups: Mapped[list["Group"]] = relationship(
        secondary = UserGroup.__tablename__,
        back_populates = "users",
    )

    __mapper_args__ = {
        "polymorphic_identity": "user",
    }


# Keep these schemas in sync with User model
from fastapi_users import schemas

class UserRead(schemas.BaseUser[uuid_UUID]):
    pass

class UserCreate(schemas.BaseUserCreate):
    pass

class UserUpdate(schemas.BaseUserUpdate):
    pass

from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, String, ForeignKey, UUID
from .Principal import Principal
from .UserGroup import UserGroup


class Group(Principal):
    __tablename__ = "groups"

    id = Column(UUID, ForeignKey("principals.id"), primary_key=True)

    name = Column(String, unique=True, nullable=False, index=True)

    users: Mapped[list["User"]] = relationship(
        "User",
        secondary=UserGroup.__tablename__,
        back_populates="groups",
    )

    __mapper_args__ = {
        "polymorphic_identity": "group",
    }

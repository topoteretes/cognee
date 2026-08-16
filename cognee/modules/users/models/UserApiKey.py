from uuid import UUID, uuid4
from typing import Optional
from sqlalchemy import ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column
from cognee.infrastructure.databases.relational.ModelBase import Base


class UserApiKey(Base):
    __tablename__ = "user_api_key"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("principals.id", ondelete="CASCADE"), nullable=False, index=True
    )
    api_key: Mapped[str] = mapped_column(nullable=False)
    label: Mapped[Optional[str]] = mapped_column(nullable=True)
    name: Mapped[Optional[str]] = mapped_column(nullable=True)

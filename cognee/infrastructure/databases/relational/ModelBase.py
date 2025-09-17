from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncAttrs


class Base(AsyncAttrs, DeclarativeBase):
    """
    Represents a base class for declarative models using SQLAlchemy.

    The Base class provides the foundation for creating ORM-mapped classes and manages the
    mapping of classes to database tables.
    """

    pass

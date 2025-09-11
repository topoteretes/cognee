from typing import Any, Callable, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from .get_async_session import get_async_session


def get_session_from_args(args):
    last_arg = args[-1]
    if isinstance(last_arg, AsyncSession):
        return last_arg
    return None


def with_async_session(func: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapper(*args, **kwargs):
        session = kwargs.get("session") or get_session_from_args(args)  # type: Optional[AsyncSession]

        if session is None:
            async with get_async_session() as session:
                result = await func(*args, **kwargs, session=session)
                await session.commit()
                return result
        else:
            return await func(*args, **kwargs)

    return wrapper

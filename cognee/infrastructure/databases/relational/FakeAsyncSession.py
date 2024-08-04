import inspect
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

class FakeAsyncSession:
    def __init__(self, session: Session):
        self.session = session

    def run_sync(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        If the method being called is async in AsyncSession, create a fake async version
        for Session so callers can `await` as usual. Think `commit`, `refresh`,
        `delete`, etc.
        """
        async_session_attr = getattr(AsyncSession, name, None)
        session_attr = getattr(self.session, name)

        if not inspect.iscoroutinefunction(async_session_attr):
            return session_attr

        async def async_wrapper(*args, **kwargs):
            return session_attr(*args, **kwargs)

        return async_wrapper

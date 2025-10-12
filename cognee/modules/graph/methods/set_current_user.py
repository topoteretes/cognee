import uuid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def set_current_user(session: AsyncSession, user_id: uuid.UUID, local: bool = False):
    scope = "LOCAL " if local else ""
    await session.execute(text(f"SET {scope}app.current_user_id = '{user_id}'"))

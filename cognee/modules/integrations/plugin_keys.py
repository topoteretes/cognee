"""API keys for plugin agents: provisioning only ever adds one.

A plugin keeps one agent sub-user per ``(user, plugin_key)``, and every machine
the plugin runs on provisions its own key for that agent. Provisioning never
deletes a key, so a new machine never logs out another one. An agent already
holding ``MAX_USER_API_KEYS`` keys gets a limit error instead; its owner frees a
slot by revoking a key they no longer need.

Adding a key and revoking all of them each run as one transaction under a lock
on the agent's user row: concurrent provisions serialize instead of racing past
the cap, and a disconnect never sees a key list that a provision is changing.
"""

import asyncio
from uuid import UUID

from sqlalchemy import delete, func, select, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.api_key.create_api_key import apiKeySettings, build_api_key_row
from cognee.modules.users.models import User
from cognee.modules.users.models.UserApiKey import UserApiKey


class PluginAgentKeyLimitError(Exception):
    """The agent already holds the maximum number of API keys."""

    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(f"Agent has reached the maximum of {limit} API keys.")


async def _lock_agent_row(session, agent_user_id: UUID) -> None:
    """Serialize key writes for one agent until the transaction ends.

    Postgres: ``SELECT … FOR UPDATE`` on the agent's user row. SQLite (and
    Turso): the driver only opens a transaction before the first write, so a
    leading no-op UPDATE of the same row is what makes this a write transaction
    from its first statement — the equivalent of ``BEGIN IMMEDIATE``. A second
    writer waits on ``busy_timeout`` and then sees the first one's commit.
    """
    if session.bind.dialect.name == "postgresql":
        await session.execute(select(User.id).where(User.id == agent_user_id).with_for_update())
        return
    await session.execute(
        update(User).where(User.id == agent_user_id).values(parent_user_id=User.parent_user_id)
    )


async def add_plugin_agent_key(agent_user_id: UUID, plugin_key: str) -> UserApiKey:
    """Add a key to the agent, atomically, unless it is already full.

    Returns the new row with the raw key on ``api_key`` (shown once).

    Raises:
        PluginAgentKeyLimitError: the agent already holds ``MAX_USER_API_KEYS``
        keys; nothing was added.
    """
    # Built before the transaction: with HASH_API_KEY on, hashing is a slow
    # PBKDF2 run that must not hold the agent's row lock (or SQLite's write
    # lock), nor block the event loop.
    user_api_key, raw_key = await asyncio.to_thread(build_api_key_row, agent_user_id, plugin_key)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        async with session.begin():
            await _lock_agent_row(session, agent_user_id)

            held = (
                await session.execute(
                    select(func.count())
                    .select_from(UserApiKey)
                    .where(UserApiKey.user_id == agent_user_id)
                )
            ).scalar_one()
            limit = apiKeySettings.max_user_api_keys
            if held + 1 > limit:
                raise PluginAgentKeyLimitError(limit)

            session.add(user_api_key)

        user_api_key.api_key = raw_key  # return raw key so caller can show it once
        return user_api_key


async def revoke_plugin_agent_keys(agent_user_id: UUID) -> int:
    """Delete every key the agent holds, in one transaction. Returns how many."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        async with session.begin():
            await _lock_agent_row(session, agent_user_id)
            result = await session.execute(
                delete(UserApiKey).where(UserApiKey.user_id == agent_user_id)
            )
        return result.rowcount

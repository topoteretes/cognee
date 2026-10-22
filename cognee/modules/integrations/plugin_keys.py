"""API keys for plugin agents: provisioning only ever adds one.

A plugin keeps one agent sub-user per ``(user, plugin_key)``, and every machine
the plugin runs on provisions its own key for that agent. Below the per-agent
cap (``MAX_USER_API_KEYS``), provisioning never deletes a key, so a new machine
never logs out another one. At the cap, the least recently used keys are
evicted, just enough for the new one to fit: in practice the keys of machines
or containers that are gone.

Every key write for an agent (add, revoke one, revoke all) is one transaction
under a lock on the agent's user row: a failure anywhere rolls an eviction back
too, concurrent provisions serialize instead of racing past the cap, and a
revoke never trips over a key another transaction just evicted.
"""

import asyncio
from uuid import UUID

from sqlalchemy import delete, func, select, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.api_key.create_api_key import apiKeySettings, build_api_key_row
from cognee.modules.users.models import User
from cognee.modules.users.models.UserApiKey import UserApiKey


class PluginAgentKeyLimitError(Exception):
    """The key cap leaves no room for a key, even after evicting every other one."""

    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(f"MAX_USER_API_KEYS={limit} leaves no room for an agent key.")


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
    """Add a key to the agent, evicting its least recently used keys only if it is full.

    Only ``agent_user_id``'s keys are touched, and only as many as the new key
    needs room for. A key never used ranks by when it was created; a legacy key
    with neither timestamp ranks first. Returns the new row with the raw key on
    ``api_key`` (shown once).

    Raises:
        PluginAgentKeyLimitError: ``MAX_USER_API_KEYS`` is below 1; nothing was
        deleted or added.
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
            overflow = held + 1 - limit
            if overflow > 0:
                if overflow > held:
                    raise PluginAgentKeyLimitError(limit)
                last_seen = func.coalesce(UserApiKey.last_used_at, UserApiKey.created_at)
                evicted = (
                    (
                        await session.execute(
                            select(UserApiKey.id)
                            .where(UserApiKey.user_id == agent_user_id)
                            .order_by(last_seen.asc().nulls_first())
                            .limit(overflow)
                        )
                    )
                    .scalars()
                    .all()
                )
                await session.execute(delete(UserApiKey).where(UserApiKey.id.in_(evicted)))

            session.add(user_api_key)

        user_api_key.api_key = raw_key  # return raw key so caller can show it once
        return user_api_key


async def revoke_plugin_agent_key(agent_user_id: UUID, key_id: UUID) -> bool:
    """Delete one of the agent's keys. False when the agent holds no key with that id."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        async with session.begin():
            await _lock_agent_row(session, agent_user_id)
            result = await session.execute(
                delete(UserApiKey).where(
                    UserApiKey.id == key_id, UserApiKey.user_id == agent_user_id
                )
            )
        return result.rowcount > 0


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

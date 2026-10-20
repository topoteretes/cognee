"""API keys for plugin agents: provisioning only ever adds one.

A plugin keeps one agent sub-user per ``(user, plugin_key)``, and every machine
the plugin runs on provisions its own key for that agent. Provisioning never
revokes a key, so a new machine never logs out another one. Keys clean
themselves up instead: each provision first prunes the agent's stale keys
(unused for ``plugin_key_idle_days``, or never used and older than
``plugin_key_unused_hours``), so the per-agent cap is rarely reached.

Prune, cap check and insert are one transaction under a lock on the agent's
user row: a failure anywhere rolls the prune back too, and two concurrent
provisions serialize instead of racing past the cap.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import and_, delete, false, func, or_, select, update

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


def _stale_key_condition(now: datetime):
    """Keys idle past ``plugin_key_idle_days``, or never used past ``plugin_key_unused_hours``.

    A key with neither timestamp (legacy rows) matches neither rule and is kept.
    """
    rules = []
    if apiKeySettings.plugin_key_idle_days > 0:
        idle_cutoff = now - timedelta(days=apiKeySettings.plugin_key_idle_days)
        rules.append(UserApiKey.last_used_at < idle_cutoff)
    if apiKeySettings.plugin_key_unused_hours > 0:
        unused_cutoff = now - timedelta(hours=apiKeySettings.plugin_key_unused_hours)
        rules.append(and_(UserApiKey.last_used_at.is_(None), UserApiKey.created_at < unused_cutoff))
    return or_(*rules) if rules else false()


async def add_plugin_agent_key(agent_user_id: UUID, plugin_key: str) -> UserApiKey:
    """Prune the agent's stale keys and add a new one, atomically.

    Only ``agent_user_id``'s keys are touched. Returns the new row with the raw
    key on ``api_key`` (shown once).

    Raises:
        PluginAgentKeyLimitError: the agent is still full after pruning; nothing
        was deleted or added.
    """
    # Built before the transaction: with HASH_API_KEY on, hashing is a slow
    # PBKDF2 run that must not hold the agent's row lock (or SQLite's write
    # lock), nor block the event loop.
    user_api_key, raw_key = await asyncio.to_thread(build_api_key_row, agent_user_id, plugin_key)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        async with session.begin():
            await _lock_agent_row(session, agent_user_id)

            await session.execute(
                delete(UserApiKey).where(
                    UserApiKey.user_id == agent_user_id,
                    _stale_key_condition(datetime.now(timezone.utc)),
                )
            )

            remaining = (
                await session.execute(
                    select(func.count())
                    .select_from(UserApiKey)
                    .where(UserApiKey.user_id == agent_user_id)
                )
            ).scalar_one()
            limit = apiKeySettings.max_user_api_keys
            if remaining + 1 > limit:
                raise PluginAgentKeyLimitError(limit)

            session.add(user_api_key)

        user_api_key.api_key = raw_key  # return raw key so caller can show it once
        return user_api_key


async def revoke_agent_api_key(agent_user_id: UUID, key_id: UUID) -> bool:
    """Delete one of the agent's keys. False when the agent holds no such key."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session, session.begin():
        revoked = await session.execute(
            delete(UserApiKey)
            .where(UserApiKey.id == key_id, UserApiKey.user_id == agent_user_id)
            .returning(UserApiKey.id)
        )
        return revoked.scalar_one_or_none() is not None

"""Live Letta API memory source."""

from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from cognee.modules.migration.cogx import COGXRecord
from cognee.modules.migration.sources.base import MemorySource
from cognee.modules.migration.sources.letta import iter_letta_records
from cognee.modules.migration.sources.live._utils import call_maybe_async

_ROLE_MAP = {
    "user_message": "user",
    "assistant_message": "assistant",
    "system_message": "system",
}


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _map_role(message: Any) -> str:
    message_type = getattr(message, "message_type", None) or getattr(message, "role", None)
    if message_type in _ROLE_MAP:
        return _ROLE_MAP[message_type]
    if isinstance(message_type, str):
        return message_type
    return "unknown"


def _block_dict(block: Any) -> Dict[str, Any]:
    return {
        "id": getattr(block, "id", None),
        "label": getattr(block, "label", None) or getattr(block, "name", None),
        "value": getattr(block, "value", None) or getattr(block, "content", None),
        "limit": getattr(block, "limit", None),
    }


def _message_dict(message: Any) -> Dict[str, Any]:
    return {
        "role": _map_role(message),
        "content": getattr(message, "content", None),
        "created_at": _isoformat(
            getattr(message, "date", None) or getattr(message, "created_at", None)
        ),
    }


def _passage_dict(passage: Any) -> Dict[str, Any]:
    return {
        "id": getattr(passage, "id", None),
        "text": getattr(passage, "text", None) or getattr(passage, "content", None),
        "created_at": _isoformat(getattr(passage, "created_at", None)),
    }


async def _collect_iterator(items: Any) -> List[Any]:
    if hasattr(items, "__aiter__"):
        return [item async for item in items]
    if hasattr(items, "__iter__") and not isinstance(items, (list, dict, str)):
        return list(items)
    return list(items or [])


async def fetch_letta_snapshot(
    client: Any, agent_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Export Letta agent state into the LettaSource agent-file dict shape."""
    from cognee.modules.migration.sources.live._utils import require_extra

    require_extra("letta", "letta_client")

    if agent_ids is None:
        agents = await _collect_iterator(await call_maybe_async(client.agents.list))
        agent_ids = [getattr(agent, "id", None) for agent in agents]
        agent_ids = [agent_id for agent_id in agent_ids if agent_id]

    exports: List[Dict[str, Any]] = []
    for agent_id in agent_ids:
        agent = await call_maybe_async(client.agents.retrieve, agent_id)
        blocks = await _collect_iterator(
            await call_maybe_async(client.agents.blocks.list, agent_id=agent_id)
        )
        messages = await _collect_iterator(
            await call_maybe_async(client.agents.messages.list, agent_id=agent_id, order="asc")
        )
        passages = await _collect_iterator(
            await call_maybe_async(client.agents.passages.list, agent_id=agent_id)
        )
        exports.append(
            {
                "name": getattr(agent, "name", None) or agent_id,
                "core_memory": [_block_dict(block) for block in blocks],
                "messages": [_message_dict(message) for message in messages],
                "archival_memory": [_passage_dict(passage) for passage in passages],
            }
        )
    return {"agents": exports}


class LettaLiveSource(MemorySource):
    """Fetch Letta agent memory via an injected Letta client.

    Args:
        client: ``letta_client.Letta`` or ``AsyncLetta`` instance.
        agent_ids: Agents to export; ``None`` exports all agents from
            ``agents.list()``.
        mode: Import mode (default ``re-derive``).
    """

    source_system = "letta"
    replayable = True

    def __init__(
        self,
        client: Any,
        agent_ids: Optional[List[str]] = None,
        mode: str = "re-derive",
    ):
        super().__init__(mode=mode)
        self._client = client
        self._agent_ids = agent_ids
        self._snapshot: Optional[Dict[str, Any]] = None

    async def _ensure_snapshot(self) -> Dict[str, Any]:
        if self._snapshot is None:
            self._snapshot = await fetch_letta_snapshot(self._client, agent_ids=self._agent_ids)
        return self._snapshot

    async def records(self) -> AsyncIterator[COGXRecord]:
        snapshot = await self._ensure_snapshot()
        async for record in iter_letta_records(snapshot, source_system=self.source_system):
            yield record

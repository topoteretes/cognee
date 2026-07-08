"""Letta (MemGPT) agent-file source.

Reads a Letta Agent File (``.af``, a JSON serialization of agents) and yields:

- core memory blocks -> :class:`COGXMemoryBlock`
- message history    -> one :class:`COGXEpisode` per agent
- archival memory    -> one :class:`COGXDocument` per passage

The parser is tolerant of key-name variations across Letta versions
(``core_memory``/``blocks``/``memory_blocks``, ``messages``/``in_context_messages``,
``archival_memory``/``passages``). Message content may be a string or a list of
typed parts; only text parts are imported.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union

from cognee.modules.migration.cogx import (
    COGXDocument,
    COGXEpisode,
    COGXMemoryBlock,
    COGXRecord,
    COGXScope,
    COGXTurn,
    parse_timestamp,
)
from cognee.modules.migration.sources.base import MemorySource


def _first_list(container: Dict[str, Any], *keys: str) -> List[Dict[str, Any]]:
    for key in keys:
        value = container.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", message.get("text"))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


class LettaSource(MemorySource):
    source_system = "letta"

    def __init__(self, data: Union[str, Path, Dict[str, Any]], mode: str = "re-derive"):
        super().__init__(mode=mode)
        self._data = data

    def _load_raw(self) -> Dict[str, Any]:
        data = self._data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Unrecognized Letta agent file: expected a JSON object.")
        return data

    async def records(self) -> AsyncIterator[COGXRecord]:
        data = self._load_raw()
        agents = _first_list(data, "agents")
        if not agents:
            # A file may serialize a single agent at the top level.
            agents = [data]

        shared_blocks = {
            str(block.get("id")): block for block in _first_list(data, "blocks") if block.get("id")
        }

        for agent_index, agent in enumerate(agents):
            agent_name = str(agent.get("name") or f"agent-{agent_index}")
            scope = COGXScope(agent_id=agent_name)

            blocks = _first_list(agent, "core_memory", "blocks", "memory_blocks")
            if not blocks and shared_blocks:
                block_ids = agent.get("block_ids") or agent.get("core_memory_block_ids") or []
                blocks = [shared_blocks[str(bid)] for bid in block_ids if str(bid) in shared_blocks]
            for block_index, block in enumerate(blocks):
                value = block.get("value") or block.get("content") or ""
                if not isinstance(value, str) or not value.strip():
                    continue
                label = str(block.get("label") or block.get("name") or f"block-{block_index}")
                yield COGXMemoryBlock(
                    external_system=self.source_system,
                    external_id=str(block.get("id") or f"{agent_name}:block:{label}"),
                    label=label,
                    value=value,
                    limit=block.get("limit"),
                    scope=scope,
                )

            messages = _first_list(agent, "messages", "in_context_messages", "message_history")
            turns = []
            for message in messages:
                text = _message_text(message)
                role = str(message.get("role") or "unknown")
                if not text.strip() or role in ("system", "tool"):
                    continue
                turns.append(
                    COGXTurn(
                        role=role,
                        content=text,
                        occurred_at=parse_timestamp(
                            message.get("created_at") or message.get("timestamp")
                        ),
                    )
                )
            if turns:
                yield COGXEpisode(
                    external_system=self.source_system,
                    external_id=f"{agent_name}:messages",
                    title=f"Conversation history of agent {agent_name}",
                    turns=turns,
                    scope=scope,
                )

            passages = _first_list(agent, "archival_memory", "passages", "archival_passages")
            for passage_index, passage in enumerate(passages):
                text = passage.get("text") or passage.get("content")
                if not isinstance(text, str) or not text.strip():
                    continue
                yield COGXDocument(
                    external_system=self.source_system,
                    external_id=str(passage.get("id") or f"{agent_name}:passage:{passage_index}"),
                    content=text,
                    created_at=parse_timestamp(passage.get("created_at")),
                    scope=scope,
                )

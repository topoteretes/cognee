"""Zep / Graphiti memory source.

Reads a JSON export of a Zep or Graphiti (OSS) knowledge graph and yields:

- episodes (verbatim ingested content) -> :class:`COGXEpisode`
- entity nodes                          -> :class:`COGXEntity`
- relation edges ("facts")              -> :class:`COGXFact` carrying the
  bi-temporal ``valid_at``/``invalid_at``/``expired_at`` fields

Expected shape (tolerant of key-name variants)::

    {
      "episodes": [{"uuid", "name", "content"|"episode_body", "created_at", ...}],
      "entities"|"nodes": [{"uuid", "name", "labels"|"label", "summary", ...}],
      "facts"|"edges": [{"uuid", "source_node_uuid", "target_node_uuid",
                          "name"|"relation", "fact", "valid_at", "invalid_at", ...}]
    }

For OSS Graphiti the export JSON can be produced with a direct Cypher dump of
EntityNode/EpisodicNode/RELATES_TO records. Defaults to ``hybrid`` mode since
Graphiti keeps both verbatim episodes and a derived graph.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union

from cognee.modules.migration.cogx import (
    COGXEntity,
    COGXEpisode,
    COGXFact,
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


class ZepSource(MemorySource):
    source_system = "zep"

    def __init__(self, data: Union[str, Path, Dict[str, Any]], mode: str = "hybrid"):
        super().__init__(mode=mode)
        self._data = data

    def _load_raw(self) -> Dict[str, Any]:
        data = self._data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Unrecognized Zep/Graphiti export: expected a JSON object.")
        return data

    async def records(self) -> AsyncIterator[COGXRecord]:
        data = self._load_raw()

        for index, episode in enumerate(_first_list(data, "episodes", "episodic_nodes")):
            content = episode.get("content") or episode.get("episode_body")
            if not isinstance(content, str) or not content.strip():
                continue
            occurred_at = parse_timestamp(episode.get("valid_at") or episode.get("created_at"))
            yield COGXEpisode(
                external_system=self.source_system,
                external_id=str(episode.get("uuid") or episode.get("id") or f"episode-{index}"),
                title=episode.get("name"),
                turns=[COGXTurn(role="episode", content=content, occurred_at=occurred_at)],
                created_at=parse_timestamp(episode.get("created_at")),
                scope=COGXScope(
                    user_id=episode.get("user_id"),
                    session_id=episode.get("group_id") or episode.get("session_id"),
                ),
                metadata=(
                    {"source_description": episode.get("source_description")}
                    if episode.get("source_description")
                    else {}
                ),
            )

        for index, node in enumerate(_first_list(data, "entities", "nodes", "entity_nodes")):
            name = node.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            labels = node.get("labels") or node.get("label") or []
            if isinstance(labels, str):
                labels = [labels]
            entity_type = next((label for label in labels if label != "Entity"), None)
            yield COGXEntity(
                external_system=self.source_system,
                external_id=str(node.get("uuid") or node.get("id") or f"entity-{index}"),
                name=name,
                entity_type=entity_type,
                description=node.get("summary") or node.get("description"),
                attributes=node.get("attributes") or {},
                created_at=parse_timestamp(node.get("created_at")),
                scope=COGXScope(session_id=node.get("group_id")),
            )

        for index, edge in enumerate(_first_list(data, "facts", "edges", "entity_edges")):
            subject_ref = edge.get("source_node_uuid") or edge.get("source")
            object_ref = edge.get("target_node_uuid") or edge.get("target")
            if not subject_ref or not object_ref:
                continue
            yield COGXFact(
                external_system=self.source_system,
                external_id=str(edge.get("uuid") or edge.get("id") or f"fact-{index}"),
                subject_ref=str(subject_ref),
                predicate=str(edge.get("name") or edge.get("relation") or "relates_to"),
                object_ref=str(object_ref),
                fact_text=edge.get("fact"),
                valid_at=parse_timestamp(edge.get("valid_at")),
                invalid_at=parse_timestamp(edge.get("invalid_at") or edge.get("expired_at")),
                created_at=parse_timestamp(edge.get("created_at")),
                provenance=[str(episode) for episode in edge.get("episodes") or []],
                scope=COGXScope(session_id=edge.get("group_id")),
            )


class GraphitiSource(ZepSource):
    """Alias for OSS Graphiti exports (same shape as Zep graph exports)."""

    source_system = "graphiti"

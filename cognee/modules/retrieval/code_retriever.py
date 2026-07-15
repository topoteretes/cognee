"""Deterministic queries over an Enola-derived Cognee code graph.

The retriever intentionally uses only the graph adapter.  It does not invoke an
LLM, an embedding engine, or vector search.  ``query`` is the human-readable
seed while ``config`` selects one exact graph operation and its arguments.

For backend portability, the retriever loads the dataset-scoped code subgraph
through the graph adapter. Parsed graph indexes are cached by dataset and graph
database identity, with bounded LRU/TTL retention and explicit invalidation by
the code-graph write pipeline. Callers should select the narrowest dataset that
contains the repositories they want to traverse.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from threading import RLock
import time
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional
from uuid import UUID

from cognee.context_global_variables import current_dataset_id
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.config import get_graph_context_config
from cognee.modules.retrieval.base_retriever import BaseRetriever


CODE_NODE_TYPES = (
    "ApiEndpoint",
    "CodeModule",
    "CodeService",
    "CodeSymbol",
    "CodeTestReference",
    "CodeFileReference",
    "ExternalDependency",
    "StorageResource",
)

_KIND_BY_TYPE = {
    "ApiEndpoint": "route",
    "CodeModule": "module",
    "CodeService": "service",
    "CodeSymbol": "symbol",
    "CodeTestReference": "test_ref",
    "CodeFileReference": "file_ref",
    "ExternalDependency": "dependency",
    "StorageResource": "storage",
}
_TYPE_BY_KIND = {kind: node_type for node_type, kind in _KIND_BY_TYPE.items()}
_OPERATIONS = {
    "query_facts",
    "explore",
    "traverse",
    "find_path",
    "impact_analysis",
}
_TYPE_SYMBOL_KINDS = {"struct", "class", "interface", "type"}
_CODE_EXTENSIONS = {
    "c",
    "cc",
    "cpp",
    "cs",
    "go",
    "h",
    "hpp",
    "java",
    "js",
    "jsx",
    "kt",
    "kts",
    "m",
    "mjs",
    "mm",
    "php",
    "py",
    "rb",
    "rs",
    "scala",
    "swift",
    "ts",
    "tsx",
}
_MISSING = object()
_CODE_GRAPH_SNAPSHOT_FORMAT_VERSION = 1
_DEFAULT_CODE_GRAPH_CACHE_MAX_ENTRIES = 16
_DEFAULT_CODE_GRAPH_CACHE_TTL_SECONDS = 60.0
_GRAPH_IDENTITY_FIELDS = (
    "graph_database_provider",
    "graph_database_url",
    "graph_database_name",
    "graph_database_username",
    "graph_database_host",
    "graph_database_port",
    "graph_database_key",
    "graph_file_path",
    "graph_dataset_database_handler",
)
_INTERNAL_EDGE_PROPERTIES = {
    "source_node_id",
    "target_node_id",
    "relationship_name",
    "relationship_type",
    "edge_text",
    "created_at",
    "updated_at",
}


class CodeSearchValidationError(CogneeValidationError):
    """Invalid deterministic CODE operation or graph-node resolution."""

    def __init__(self, message: str):
        super().__init__(message=message, name="CodeSearchValidationError", log=False)


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_value(value: Any) -> Any:
    """Convert backend-specific UUID/JSON containers into plain JSON values."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = [_normalize_value(item) for item in value]
        return sorted(items, key=_json_key) if isinstance(value, set) else items
    if hasattr(value, "model_dump"):
        return _normalize_value(value.model_dump())
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _normalize_value(json.loads(stripped))
            except (TypeError, ValueError):
                pass
    return value


def _normalize_properties(value: Any) -> dict[str, Any]:
    """Flatten adapters which return user properties under ``properties``."""
    normalized = _normalize_value(value)
    if not isinstance(normalized, dict):
        return {}
    nested = normalized.get("properties")
    if isinstance(nested, dict):
        normalized = {
            **nested,
            **{key: val for key, val in normalized.items() if key != "properties"},
        }
    return normalized


def _as_strings(value: Any, field: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple, set)):
        raise CodeSearchValidationError(f"CODE {field} must be a string or a list of strings.")
    if not all(isinstance(item, str) and item for item in value):
        raise CodeSearchValidationError(f"CODE {field} must contain only non-empty strings.")
    return sorted(set(value)) if isinstance(value, set) else list(dict.fromkeys(value))


def _bounded_int(value: Any, *, field: str, default: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodeSearchValidationError(f"CODE {field} must be an integer.")
    if value <= 0:
        return default
    return min(value, maximum)


def _offset(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CodeSearchValidationError("CODE offset must be a non-negative integer.")
    return value


def _direction(value: Any, *, default: str) -> str:
    direction = str(value or default).lower()
    if direction not in {"forward", "reverse", "both"}:
        raise CodeSearchValidationError("CODE direction must be 'forward', 'reverse', or 'both'.")
    return direction


def _node_type(value: str) -> str:
    if value in CODE_NODE_TYPES:
        return value
    normalized = value.strip().lower()
    if normalized in _TYPE_BY_KIND:
        return _TYPE_BY_KIND[normalized]
    for candidate in CODE_NODE_TYPES:
        if candidate.lower() == normalized:
            return candidate
    raise CodeSearchValidationError(
        f"Unsupported CODE node kind/type {value!r}; expected one of "
        f"{', '.join(sorted(_TYPE_BY_KIND))}."
    )


def _node_types(value: Any, field: str = "node_types") -> set[str] | None:
    values = _as_strings(value, field)
    return {_node_type(item) for item in values} if values else None


def _scalar_text(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return _json_key(value)
    return str(value)


class _CodeGraphSnapshot:
    """Stable in-memory indexes and adjacency over one dataset-scoped graph."""

    def __init__(self, raw_nodes: Iterable[Any], raw_edges: Iterable[Any]):
        self.nodes: dict[str, dict[str, Any]] = {}
        for item in raw_nodes or []:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            raw_id, raw_properties = item
            node_id = str(raw_id)
            properties = _normalize_properties(raw_properties)
            node_type = properties.get("type")
            if node_type not in CODE_NODE_TYPES:
                continue
            properties["id"] = node_id
            self.nodes[node_id] = properties

        normalized_edges = []
        for item in raw_edges or []:
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                continue
            source_id, target_id, relationship = map(str, item[:3])
            if source_id not in self.nodes or target_id not in self.nodes:
                continue
            properties = _normalize_properties(item[3] if len(item) > 3 else {})
            normalized_edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": relationship,
                    "properties": properties,
                }
            )

        normalized_edges.sort(
            key=lambda edge: (
                edge["source_id"],
                edge["target_id"],
                edge["type"],
                _json_key(edge["properties"]),
            )
        )
        self.edges = []
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in normalized_edges:
            identity = (edge["source_id"], edge["target_id"], edge["type"])
            if identity in seen_edges:
                continue
            seen_edges.add(identity)
            self.edges.append(edge)

        self.forward: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.reverse: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in self.edges:
            self.forward[edge["source_id"]].append(edge)
            self.reverse[edge["target_id"]].append(edge)
        for adjacency in (self.forward, self.reverse):
            for node_id in adjacency:
                adjacency[node_id].sort(key=self.edge_sort_key)

        self.by_type: dict[str, set[str]] = defaultdict(set)
        self.by_name: dict[str, set[str]] = defaultdict(set)
        self.by_file: dict[str, set[str]] = defaultdict(set)
        self.by_repo: dict[str, set[str]] = defaultdict(set)
        self.by_property: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for node_id, node in self.nodes.items():
            self.by_type[str(node.get("type", ""))].add(node_id)
            self.by_name[str(node.get("name", ""))].add(node_id)
            self.by_file[self.file_of(node)].add(node_id)
            self.by_repo[str(node.get("repo", ""))].add(node_id)
            for key, value in node.items():
                if key != "fact_properties":
                    self._index_property(node_id, key, value)
            for key, value in self.properties_of(node).items():
                self._index_property(node_id, key, value)

    def _index_property(self, node_id: str, name: str, value: Any) -> None:
        self.by_property[name][_scalar_text(value)].add(node_id)
        if isinstance(value, Mapping):
            for child_name, child_value in value.items():
                self._index_property(node_id, f"{name}.{child_name}", child_value)

    @staticmethod
    def file_of(node: Mapping[str, Any]) -> str:
        return str(node.get("file_path") or node.get("file") or "")

    @staticmethod
    def properties_of(node: Mapping[str, Any]) -> dict[str, Any]:
        properties = node.get("fact_properties")
        return properties if isinstance(properties, dict) else {}

    def node_sort_key(self, node_id: str) -> tuple[str, ...]:
        node = self.nodes[node_id]
        return (
            str(node.get("repo", "")),
            str(node.get("type", "")),
            str(node.get("name", "")).casefold(),
            self.file_of(node),
            str(node.get("line", "")),
            node_id,
        )

    def edge_sort_key(self, edge: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            str(edge["type"]),
            self.node_sort_key(str(edge["source_id"])),
            self.node_sort_key(str(edge["target_id"])),
        )

    def edge_payload(self, edge: Mapping[str, Any]) -> dict[str, Any]:
        source = self.nodes[str(edge["source_id"])]
        target = self.nodes[str(edge["target_id"])]
        result = {
            "source_id": str(edge["source_id"]),
            "source": source.get("name"),
            "target_id": str(edge["target_id"]),
            "target": target.get("name"),
            "type": edge["type"],
        }
        public_properties = {
            key: value
            for key, value in (edge.get("properties") or {}).items()
            if key not in _INTERNAL_EDGE_PROPERTIES
        }
        if public_properties:
            result["properties"] = _normalize_value(public_properties)
        return result

    def fact(
        self,
        node_id: str,
        *,
        depth: Optional[int] = None,
        include_relations: bool = True,
    ) -> dict[str, Any]:
        node = self.nodes[node_id]
        result = {
            "id": node_id,
            "kind": _KIND_BY_TYPE[str(node["type"])],
            "type": node["type"],
            "name": node.get("name"),
        }
        aliases = {
            "file": self.file_of(node),
            "line": node.get("line"),
            "repo": node.get("repo"),
            "path": node.get("path"),
            "description": node.get("description"),
            "symbol_kind": node.get("symbol_kind"),
        }
        result.update({key: value for key, value in aliases.items() if value not in (None, "")})
        properties = self.properties_of(node)
        if properties:
            # Results are caller-owned. Returning a nested mapping from the
            # cached snapshot would let one response corrupt later queries.
            result["properties"] = _normalize_value(properties)
        if include_relations:
            relations = []
            for edge in self.forward.get(node_id, []):
                target_id = edge["target_id"]
                relations.append(
                    {
                        "type": edge["type"],
                        "target_id": target_id,
                        "target": self.nodes[target_id].get("name"),
                    }
                )
            if relations:
                result["relations"] = relations
        if depth is not None:
            result["depth"] = depth
        return result

    def adjacent(
        self, node_id: str, direction: str, relation_types: set[str] | None
    ) -> list[tuple[str, dict[str, Any]]]:
        adjacent = []
        if direction in {"forward", "both"}:
            adjacent.extend((edge["target_id"], edge) for edge in self.forward.get(node_id, []))
        if direction in {"reverse", "both"}:
            adjacent.extend((edge["source_id"], edge) for edge in self.reverse.get(node_id, []))
        if relation_types is not None:
            adjacent = [item for item in adjacent if item[1]["type"] in relation_types]
        adjacent.sort(key=lambda item: (self.node_sort_key(item[0]), self.edge_sort_key(item[1])))
        return adjacent

    def resolve(
        self,
        *,
        name: Optional[str] = None,
        node_id: Optional[str] = None,
        repo: Optional[str] = None,
        role: str = "node",
    ) -> str:
        if node_id:
            normalized_id = str(node_id)
            if normalized_id not in self.nodes:
                raise CodeSearchValidationError(
                    f"CODE could not resolve {role} id {normalized_id!r}."
                )
            if repo and self.nodes[normalized_id].get("repo") != repo:
                raise CodeSearchValidationError(
                    f"CODE {role} id {normalized_id!r} is not in repo {repo!r}."
                )
            return normalized_id

        term = str(name or "").strip()
        if not term:
            raise CodeSearchValidationError(f"CODE {role} requires a non-empty name or id.")
        lowered = term.casefold()
        ranked: list[tuple[int, str]] = []
        for candidate_id, node in self.nodes.items():
            candidate_name = str(node.get("name") or "")
            if repo and node.get("repo") != repo:
                continue
            tier = _match_tier(candidate_name, lowered)
            if tier >= 0:
                ranked.append((tier, candidate_id))
        if not ranked:
            raise CodeSearchValidationError(f"CODE could not resolve {role} name {term!r}.")
        best_tier = max(tier for tier, _ in ranked)
        best = sorted(
            (candidate_id for tier, candidate_id in ranked if tier == best_tier),
            key=self.node_sort_key,
        )
        if len(best) > 1:
            choices = ", ".join(
                f"{self.nodes[item].get('name')} [{self.nodes[item].get('type')}, "
                f"repo={self.nodes[item].get('repo') or '-'}, id={item}]"
                for item in best[:10]
            )
            raise CodeSearchValidationError(
                f"CODE {role} name {term!r} is ambiguous; use an exact id or repo. "
                f"Candidates: {choices}"
            )
        return best[0]


@dataclass(frozen=True)
class _CodeGraphSnapshotCacheKey:
    """Non-secret identity for one dataset-scoped graph snapshot."""

    format_version: int
    dataset_id: str
    database_fingerprint: str


@dataclass
class _CodeGraphSnapshotCacheEntry:
    snapshot: _CodeGraphSnapshot
    expires_at: float


class _CodeGraphSnapshotCache:
    """Bounded, concurrency-safe cache of parsed code-graph indexes.

    Loads are single-flight per key and generation. Invalidation advances the
    generation before dropping an entry, so a graph read which overlaps a write
    cannot republish (or return) its stale snapshot after that write completes.
    """

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("CODE graph cache max_entries must be at least 1.")
        if ttl_seconds <= 0:
            raise ValueError("CODE graph cache ttl_seconds must be positive.")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[_CodeGraphSnapshotCacheKey, _CodeGraphSnapshotCacheEntry] = (
            OrderedDict()
        )
        self._epochs: dict[_CodeGraphSnapshotCacheKey, int] = {}
        self._inflight: dict[
            tuple[_CodeGraphSnapshotCacheKey, int],
            asyncio.Task[tuple[_CodeGraphSnapshot, int]],
        ] = {}
        self._waiters: dict[tuple[_CodeGraphSnapshotCacheKey, int], int] = defaultdict(int)
        self._lock = RLock()

    async def get_or_load(
        self,
        key: _CodeGraphSnapshotCacheKey,
        loader: Callable[[], Awaitable[_CodeGraphSnapshot]],
    ) -> _CodeGraphSnapshot:
        while True:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None and entry.expires_at > self._clock():
                    self._entries.move_to_end(key)
                    return entry.snapshot
                if entry is not None:
                    self._entries.pop(key, None)

                epoch = self._epochs.setdefault(key, 0)
                flight_key = (key, epoch)
                task = self._inflight.get(flight_key)
                if task is None:
                    task = asyncio.create_task(self._load_and_store(key, epoch, loader))
                    self._inflight[flight_key] = task
                self._waiters[flight_key] += 1

            try:
                snapshot, loaded_epoch = await asyncio.shield(task)
            except asyncio.CancelledError:
                self._remove_waiter(flight_key)
                raise
            except BaseException:
                with self._lock:
                    still_current = self._epochs.get(key, 0) == epoch
                self._remove_waiter(flight_key)
                if not still_current:
                    continue
                raise

            with self._lock:
                still_current = self._epochs.get(key, 0) == loaded_epoch
            self._remove_waiter(flight_key)
            if still_current:
                return snapshot

    async def _load_and_store(
        self,
        key: _CodeGraphSnapshotCacheKey,
        epoch: int,
        loader: Callable[[], Awaitable[_CodeGraphSnapshot]],
    ) -> tuple[_CodeGraphSnapshot, int]:
        flight_key = (key, epoch)
        try:
            snapshot = await loader()
        except BaseException:
            with self._lock:
                self._inflight.pop(flight_key, None)
                self._prune_epoch_if_idle(key)
            raise

        with self._lock:
            self._inflight.pop(flight_key, None)
            if self._epochs.get(key, 0) == epoch:
                self._entries[key] = _CodeGraphSnapshotCacheEntry(
                    snapshot=snapshot,
                    expires_at=self._clock() + self.ttl_seconds,
                )
                self._entries.move_to_end(key)
                self._evict_lru_entries()
            self._prune_epoch_if_idle(key)
        return snapshot, epoch

    def invalidate(self, key: _CodeGraphSnapshotCacheKey) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._epochs[key] = self._epochs.get(key, 0) + 1
            self._prune_epoch_if_idle(key)

    def clear(self) -> None:
        """Invalidate all entries without cancelling in-progress graph reads."""
        with self._lock:
            keys = set(self._entries) | set(self._epochs)
            keys.update(key for key, _epoch in self._inflight)
            keys.update(key for key, _epoch in self._waiters)
            self._entries.clear()
            for key in keys:
                self._epochs[key] = self._epochs.get(key, 0) + 1
            for key in keys:
                self._prune_epoch_if_idle(key)

    def contains(self, key: _CodeGraphSnapshotCacheKey) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and entry.expires_at > self._clock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _remove_waiter(self, flight_key: tuple[_CodeGraphSnapshotCacheKey, int]) -> None:
        key, _epoch = flight_key
        with self._lock:
            remaining = self._waiters.get(flight_key, 0) - 1
            if remaining > 0:
                self._waiters[flight_key] = remaining
            else:
                self._waiters.pop(flight_key, None)
            self._prune_epoch_if_idle(key)

    def _evict_lru_entries(self) -> None:
        while len(self._entries) > self.max_entries:
            evicted_key, _entry = self._entries.popitem(last=False)
            self._prune_epoch_if_idle(evicted_key)

    def _prune_epoch_if_idle(self, key: _CodeGraphSnapshotCacheKey) -> None:
        if key in self._entries:
            return
        if any(inflight_key == key for inflight_key, _epoch in self._inflight):
            return
        if any(waiter_key == key for waiter_key, _epoch in self._waiters):
            return
        self._epochs.pop(key, None)


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _code_graph_snapshot_cache_key(
    *,
    dataset_id: Any = _MISSING,
    graph_config: Optional[Mapping[str, Any]] = None,
) -> _CodeGraphSnapshotCacheKey:
    """Build a stable key without retaining database credentials in memory."""
    if dataset_id is _MISSING:
        dataset_id = current_dataset_id.get()
    config = dict(get_graph_context_config() if graph_config is None else graph_config)
    database_identity = {
        field: _normalize_value(config.get(field)) for field in _GRAPH_IDENTITY_FIELDS
    }
    fingerprint = hashlib.sha256(_json_key(database_identity).encode("utf-8")).hexdigest()
    return _CodeGraphSnapshotCacheKey(
        format_version=_CODE_GRAPH_SNAPSHOT_FORMAT_VERSION,
        dataset_id=str(dataset_id or ""),
        database_fingerprint=fingerprint,
    )


_CODE_GRAPH_SNAPSHOT_CACHE = _CodeGraphSnapshotCache(
    max_entries=_positive_int_env(
        "CODE_GRAPH_CACHE_MAX_ENTRIES", _DEFAULT_CODE_GRAPH_CACHE_MAX_ENTRIES
    ),
    ttl_seconds=_positive_float_env(
        "CODE_GRAPH_CACHE_TTL_SECONDS", _DEFAULT_CODE_GRAPH_CACHE_TTL_SECONDS
    ),
)


def invalidate_code_graph_snapshot_cache(
    *,
    all_entries: bool = False,
    dataset_id: Any = _MISSING,
    graph_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Invalidate parsed CODE indexes after graph mutations.

    The default invalidates only the current dataset/database identity. Use
    ``all_entries=True`` for administrative teardown and tests.
    """
    if all_entries:
        _CODE_GRAPH_SNAPSHOT_CACHE.clear()
        return
    _CODE_GRAPH_SNAPSHOT_CACHE.invalidate(
        _code_graph_snapshot_cache_key(dataset_id=dataset_id, graph_config=graph_config)
    )


def _short_names(name: str) -> set[str]:
    lowered = name.casefold()
    forms = set()
    if "." in lowered:
        forms.add(lowered.rsplit(".", 1)[1])
    basename = lowered.rsplit("/", 1)[-1]
    if basename != lowered:
        forms.add(basename)
    if "." in basename:
        stem, extension = basename.rsplit(".", 1)
        if extension in _CODE_EXTENSIONS:
            forms.add(stem)
    return forms


def _match_tier(name: str, lowered_term: str) -> int:
    lowered_name = name.casefold()
    if lowered_name == lowered_term:
        return 3
    if lowered_term in _short_names(name):
        return 2
    if lowered_name.endswith(lowered_term) and len(lowered_name) > len(lowered_term):
        boundary = lowered_name[-len(lowered_term) - 1]
        if boundary in {".", "/"}:
            return 2
    if lowered_term in lowered_name:
        return 1
    return -1


class CodeRetriever(BaseRetriever):
    """Run one exact code-graph operation without model inference."""

    supports_session_turn_preparation = False

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        retriever_specific_config: Optional[Mapping[str, Any]] = None,
        **operation_config: Any,
    ):
        combined = dict(retriever_specific_config or {})
        combined.update(dict(config or {}))
        combined.update(operation_config)
        operation = str(combined.get("operation") or "explore").lower()
        if operation not in _OPERATIONS:
            raise CodeSearchValidationError(
                f"Unsupported CODE operation {operation!r}; expected one of "
                f"{', '.join(sorted(_OPERATIONS))}."
            )
        self.operation = operation
        self.config = combined

    async def prepare_session_turn_for_retrieval(self, query: str):
        """Bypass session interpretation, which may use LLMs or vector search."""
        return SimpleNamespace(
            should_answer=True,
            response_to_user=None,
            effective_query=query or "",
            analysis=None,
            accepted_context_ids=[],
            previous_qa_id=None,
        )

    async def _snapshot(self) -> _CodeGraphSnapshot:
        key = _code_graph_snapshot_cache_key()

        async def load() -> _CodeGraphSnapshot:
            graph_engine = await get_graph_engine()
            nodes, edges = await graph_engine.get_filtered_graph_data(
                [{"type": list(CODE_NODE_TYPES)}]
            )
            return _CodeGraphSnapshot(nodes, edges)

        return await _CODE_GRAPH_SNAPSHOT_CACHE.get_or_load(key, load)

    async def get_retrieved_objects(self, query: Optional[str], query_batch=None) -> dict[str, Any]:
        if query_batch is not None:
            raise CodeSearchValidationError("SearchType.CODE does not support batched queries.")
        snapshot = await self._snapshot()
        handler = getattr(self, f"_{self.operation}")
        return handler(snapshot, query or "")

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch=None,
        retrieved_objects: Any = None,
    ) -> str:
        return _json_key(retrieved_objects)

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch=None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        return retrieved_objects

    def _query_facts(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        kinds = _as_strings(self.config.get("kinds"), "kinds")
        if self.config.get("kind"):
            kinds.append(str(self.config["kind"]))
        type_filter = {_node_type(kind) for kind in kinds} if kinds else None

        files = set(_as_strings(self.config.get("files"), "files"))
        if self.config.get("file"):
            files.add(str(self.config["file"]))
        names = set(_as_strings(self.config.get("names"), "names"))
        # query_text is a convenient shorthand only for an otherwise unconfigured
        # query_facts call. API/tool callers often must send a generic non-empty
        # query even when their structured filters (including pagination) are
        # complete; never let that text silently narrow a structured request.
        query_name = query if set(self.config).issubset({"operation"}) else ""
        substring = str(self.config.get("name") or query_name).strip().casefold()
        file_prefix = str(self.config.get("file_prefix") or "")
        repo = self.config.get("repo")
        relation_types = self._relation_types()
        property_name = self.config.get("property") or self.config.get("prop")
        if "property_value" in self.config:
            expected_property = self.config["property_value"]
        elif "prop_value" in self.config:
            expected_property = self.config["prop_value"]
        else:
            expected_property = _MISSING

        candidate_ids: set[str] | None = None
        indexed_sets = []
        if type_filter:
            indexed_sets.append(set().union(*(graph.by_type[item] for item in type_filter)))
        if files:
            indexed_sets.append(set().union(*(graph.by_file[item] for item in files)))
        if names and not substring:
            indexed_sets.append(set().union(*(graph.by_name[item] for item in names)))
        if repo is not None:
            indexed_sets.append(set(graph.by_repo[str(repo)]))
        if property_name:
            property_buckets = graph.by_property.get(str(property_name), {})
            if expected_property is _MISSING:
                indexed_sets.append(
                    set().union(*property_buckets.values()) if property_buckets else set()
                )
            else:
                indexed_sets.append(
                    set(property_buckets.get(_scalar_text(expected_property), set()))
                )
        for indexed in indexed_sets:
            candidate_ids = indexed if candidate_ids is None else candidate_ids & indexed
        if candidate_ids is None:
            candidate_ids = set(graph.nodes)

        matched = []
        for node_id in sorted(candidate_ids, key=graph.node_sort_key):
            node = graph.nodes[node_id]
            if type_filter and node.get("type") not in type_filter:
                continue
            file_name = graph.file_of(node)
            if files or file_prefix:
                if file_name not in files and not (
                    file_prefix and file_name.startswith(file_prefix)
                ):
                    continue
            node_name = str(node.get("name") or "")
            if names or substring:
                if node_name not in names and not (substring and substring in node_name.casefold()):
                    continue
            if repo is not None and node.get("repo") != repo:
                continue
            if relation_types and not any(
                edge["type"] in relation_types for edge in graph.forward.get(node_id, [])
            ):
                continue
            if property_name and not self._matches_property(
                node, str(property_name), expected_property
            ):
                continue
            matched.append(node_id)

        total = len(matched)
        offset = _offset(self.config.get("offset"))
        limit = _bounded_int(self.config.get("limit"), field="limit", default=100, maximum=500)
        page = matched[offset : offset + limit]
        return {
            "operation": "query_facts",
            "facts": [graph.fact(node_id) for node_id in page],
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < total,
        }

    @staticmethod
    def _matches_property(
        node: Mapping[str, Any], property_name: str, expected: Any = _MISSING
    ) -> bool:
        value: Any = node
        found = True
        for part in property_name.split("."):
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            else:
                found = False
                break
        if not found:
            value = _CodeGraphSnapshot.properties_of(node)
            for part in property_name.split("."):
                if isinstance(value, Mapping) and part in value:
                    value = value[part]
                else:
                    return False
        return expected is _MISSING or _scalar_text(value) == _scalar_text(expected)

    def _explore(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        node_id = graph.resolve(
            name=self.config.get("name") or self.config.get("focus") or query,
            node_id=self.config.get("id") or self.config.get("node_id"),
            repo=self.config.get("repo"),
            role="explore seed",
        )
        traversal = self._traverse_graph(
            graph,
            [node_id],
            direction=_direction(self.config.get("direction"), default="both"),
            relation_types=self._relation_types(),
            node_types=_node_types(self.config.get("node_types")),
            max_depth=_bounded_int(
                self.config.get("max_depth"), field="max_depth", default=1, maximum=2
            ),
            max_nodes=_bounded_int(
                self.config.get("max_nodes"), field="max_nodes", default=100, maximum=500
            ),
        )
        return {"operation": "explore", "focus": graph.fact(node_id), **traversal}

    def _traverse(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        seeds = self._resolve_seeds(
            graph,
            query,
            role="traverse seed",
            name_aliases=("start", "starts"),
            id_aliases=("start_id", "start_ids"),
        )
        direction = _direction(self.config.get("direction"), default="forward")
        if direction == "reverse" and self.config.get("type_rollup", True):
            seeds = self._impact_seeds(graph, seeds)
        traversal = self._traverse_graph(
            graph,
            seeds,
            direction=direction,
            relation_types=self._relation_types(),
            node_types=_node_types(self.config.get("node_types")),
            max_depth=_bounded_int(
                self.config.get("max_depth"), field="max_depth", default=5, maximum=20
            ),
            max_nodes=_bounded_int(
                self.config.get("max_nodes"), field="max_nodes", default=100, maximum=500
            ),
        )
        return {"operation": "traverse", **traversal}

    def _find_path(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        source_id = graph.resolve(
            name=(
                self.config.get("source")
                or self.config.get("from")
                or self.config.get("name")
                or query
            ),
            node_id=(
                self.config.get("source_id") or self.config.get("from_id") or self.config.get("id")
            ),
            repo=self.config.get("source_repo") or self.config.get("repo"),
            role="path source",
        )
        target_id = graph.resolve(
            name=self.config.get("target") or self.config.get("to"),
            node_id=self.config.get("target_id") or self.config.get("to_id"),
            repo=self.config.get("target_repo"),
            role="path target",
        )
        max_depth = _bounded_int(
            self.config.get("max_depth"), field="max_depth", default=10, maximum=20
        )
        relation_types = self._relation_types()
        target_candidates = (
            self._impact_seeds(graph, [target_id])
            if self.config.get("target_rollup", True)
            else [target_id]
        )
        target_candidate_set = set(target_candidates)

        if source_id in target_candidate_set:
            result = {
                "operation": "find_path",
                "from": graph.fact(source_id, include_relations=False),
                "to": graph.fact(target_id, include_relations=False),
                "found": True,
                "path": [graph.fact(source_id, depth=0, include_relations=False)],
                "edges": [],
            }
            if source_id != target_id:
                result["matched_to"] = graph.fact(source_id, include_relations=False)
            return result

        parent: dict[str, tuple[str, dict[str, Any]]] = {}
        visited = {source_id}
        queue = deque([(source_id, 0)])
        matched_target_id: Optional[str] = None
        while queue and matched_target_id is None:
            node_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor_id, edge in graph.adjacent(node_id, "forward", relation_types):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                parent[neighbor_id] = (node_id, edge)
                if neighbor_id in target_candidate_set:
                    matched_target_id = neighbor_id
                    break
                queue.append((neighbor_id, depth + 1))

        path_ids = []
        path_edges = []
        if matched_target_id is not None:
            current = matched_target_id
            while current != source_id:
                path_ids.append(current)
                previous, edge = parent[current]
                path_edges.append(edge)
                current = previous
            path_ids.append(source_id)
            path_ids.reverse()
            path_edges.reverse()
        result = {
            "operation": "find_path",
            "from": graph.fact(source_id, include_relations=False),
            "to": graph.fact(target_id, include_relations=False),
            "found": matched_target_id is not None,
            "path": [
                graph.fact(node_id, depth=depth, include_relations=False)
                for depth, node_id in enumerate(path_ids)
            ],
            "edges": [graph.edge_payload(edge) for edge in path_edges],
        }
        if matched_target_id is not None and matched_target_id != target_id:
            result["matched_to"] = graph.fact(matched_target_id, include_relations=False)
        return result

    def _impact_analysis(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        targets = self._resolve_seeds(
            graph,
            query,
            role="impact target",
            name_aliases=("target", "targets"),
            id_aliases=("target_id", "target_ids"),
        )
        impact_seeds = self._impact_seeds(graph, targets)
        max_depth = _bounded_int(
            self.config.get("max_depth"), field="max_depth", default=3, maximum=10
        )
        max_nodes = _bounded_int(
            self.config.get("max_nodes"), field="max_nodes", default=200, maximum=500
        )
        reverse = self._traverse_graph(
            graph,
            impact_seeds,
            direction="reverse",
            relation_types=self._relation_types(),
            node_types=_node_types(self.config.get("node_types")),
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        by_depth: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seed_set = set(impact_seeds)
        for node in reverse["nodes"]:
            if node["id"] not in seed_set and node["depth"] > 0:
                by_depth[str(node["depth"])].append(node)
        by_depth = dict(sorted(by_depth.items(), key=lambda item: int(item[0])))
        total_dependents = self._reachable_count(
            graph,
            impact_seeds,
            direction="reverse",
            relation_types=self._relation_types(),
            max_depth=max_depth,
        )

        target_repos = {str(graph.nodes[node_id].get("repo") or "") for node_id in targets}
        cross_repo = sorted(
            {
                str(node.get("repo"))
                for nodes in by_depth.values()
                for node in nodes
                if node.get("repo") and str(node.get("repo")) not in target_repos
            }
        )
        result = {
            "operation": "impact_analysis",
            "targets": [graph.fact(node_id, include_relations=False) for node_id in targets],
            "impact_seeds": [
                graph.fact(node_id, depth=0, include_relations=False) for node_id in impact_seeds
            ],
            "by_depth": by_depth,
            "edges": reverse["edges"],
            "total_dependents": total_dependents,
            "cross_repo_impact": cross_repo,
            "summary": self._impact_summary(by_depth, total_dependents),
            "stats": reverse["stats"],
        }
        if self.config.get("include_dependencies", False) or self.config.get(
            "include_forward", False
        ):
            result["forward_dependencies"] = self._traverse_graph(
                graph,
                targets,
                direction="forward",
                relation_types=self._relation_types(),
                node_types=_node_types(self.config.get("node_types")),
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        return result

    def _resolve_seeds(
        self,
        graph: _CodeGraphSnapshot,
        query: str,
        *,
        role: str,
        name_aliases: tuple[str, ...] = (),
        id_aliases: tuple[str, ...] = (),
    ) -> list[str]:
        ids = _as_strings(self.config.get("node_ids"), "node_ids")
        if self.config.get("id"):
            ids.append(str(self.config["id"]))
        for field in id_aliases:
            ids.extend(_as_strings(self.config.get(field), field))
        names = _as_strings(self.config.get("names"), "names")
        if self.config.get("name"):
            names.append(str(self.config["name"]))
        for field in name_aliases:
            names.extend(_as_strings(self.config.get(field), field))
        if not ids and not names and query.strip():
            names.append(query.strip())
        resolved = [
            graph.resolve(node_id=node_id, repo=self.config.get("repo"), role=role)
            for node_id in ids
        ]
        resolved.extend(
            graph.resolve(name=name, repo=self.config.get("repo"), role=role) for name in names
        )
        if not resolved:
            raise CodeSearchValidationError(f"CODE {role} requires query_text, names, or node_ids.")
        return sorted(set(resolved), key=graph.node_sort_key)

    def _relation_types(self) -> set[str] | None:
        values = []
        for field in ("relation_types", "relation_kinds", "relation"):
            values.extend(_as_strings(self.config.get(field), field))
        return set(values) if values else None

    @staticmethod
    def _traverse_graph(
        graph: _CodeGraphSnapshot,
        seeds: list[str],
        *,
        direction: str,
        relation_types: set[str] | None,
        node_types: set[str] | None,
        max_depth: int,
        max_nodes: int,
    ) -> dict[str, Any]:
        seeds = sorted(set(seeds), key=graph.node_sort_key)
        visited = set(seeds)
        included = set(seeds)
        nodes = [graph.fact(node_id, depth=0, include_relations=False) for node_id in seeds]
        queue = deque((node_id, 0) for node_id in seeds)
        traversed: dict[tuple[str, str, str], dict[str, Any]] = {}
        max_depth_reached = 0
        truncated = len(nodes) > max_nodes

        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor_id, edge in graph.adjacent(node_id, direction, relation_types):
                edge_key = (edge["source_id"], edge["target_id"], edge["type"])
                traversed.setdefault(edge_key, edge)
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                neighbor_depth = depth + 1
                max_depth_reached = max(max_depth_reached, neighbor_depth)
                if (
                    node_types is not None
                    and graph.nodes[neighbor_id].get("type") not in node_types
                ):
                    queue.append((neighbor_id, neighbor_depth))
                    continue
                if len(nodes) >= max_nodes:
                    truncated = True
                    continue
                included.add(neighbor_id)
                nodes.append(
                    graph.fact(
                        neighbor_id,
                        depth=neighbor_depth,
                        include_relations=False,
                    )
                )
                queue.append((neighbor_id, neighbor_depth))

        edges = [
            graph.edge_payload(edge)
            for edge in sorted(traversed.values(), key=graph.edge_sort_key)
            if edge["source_id"] in included and edge["target_id"] in included
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "nodes_visited": len(visited),
                "edges_traversed": len(traversed),
                "max_depth_reached": max_depth_reached,
                "truncated": truncated,
            },
        }

    @staticmethod
    def _reachable_count(
        graph: _CodeGraphSnapshot,
        seeds: list[str],
        *,
        direction: str,
        relation_types: set[str] | None,
        max_depth: int,
    ) -> int:
        seed_set = set(seeds)
        visited = set(seeds)
        queue = deque((node_id, 0) for node_id in seeds)
        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor_id, _edge in graph.adjacent(node_id, direction, relation_types):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, depth + 1))
        return len(visited - seed_set)

    @staticmethod
    def _impact_seeds(graph: _CodeGraphSnapshot, targets: list[str]) -> list[str]:
        seeds = set(targets)
        for target_id in targets:
            node = graph.nodes[target_id]
            if (
                node.get("type") != "CodeSymbol"
                or node.get("symbol_kind") not in _TYPE_SYMBOL_KINDS
            ):
                continue
            for edge in graph.forward.get(target_id, []):
                if edge["type"] == "has_method":
                    seeds.add(edge["target_id"])

            name = str(node.get("name") or "")
            if "." in name:
                package, short_name = name.rsplit(".", 1)
                constructor = f"{package}.New{short_name}"
            else:
                package = graph.file_of(node).rsplit("/", 1)[0]
                short_name = name
                constructor = f"New{short_name}"
            for candidate_id in graph.by_name.get(constructor, set()):
                candidate = graph.nodes[candidate_id]
                candidate_file = graph.file_of(candidate)
                candidate_package = (
                    str(candidate.get("name") or "").rsplit(".", 1)[0]
                    if "." in str(candidate.get("name") or "")
                    else candidate_file.rsplit("/", 1)[0]
                )
                if candidate.get("repo") == node.get("repo") and candidate_package == package:
                    seeds.add(candidate_id)
        return sorted(seeds, key=graph.node_sort_key)

    @staticmethod
    def _impact_summary(by_depth: Mapping[str, list[dict[str, Any]]], total: int) -> str:
        if total == 0:
            return "No dependents found."
        shown = sum(len(nodes) for nodes in by_depth.values())
        groups = []
        for depth, nodes in by_depth.items():
            counts: dict[str, int] = defaultdict(int)
            for node in nodes:
                counts[str(node.get("kind") or "unknown")] += 1
            detail = ", ".join(
                f"{count} {kind}{'s' if count != 1 else ''}"
                for kind, count in sorted(counts.items())
            )
            groups.append(f"depth {depth}: {detail}")
        prefix = f"{total} total dependents"
        if shown < total:
            prefix += f" (showing {shown})"
        return prefix + (" — " + "; ".join(groups) if groups else "")

"""Deterministic queries over an Enola-derived Cognee code graph.

The retriever intentionally uses only the graph adapter.  It does not invoke an
LLM, an embedding engine, or vector search.  ``query`` is the human-readable
seed while ``config`` selects one exact graph operation and its arguments.
The text-driven operations (``search``, ``ask``) rank facts with an in-memory
BM25 index over names, docstrings and paths (code_search_index.py), built on
first use and kept with the cached snapshot.

For backend portability, the retriever loads the dataset-scoped code subgraph
through the graph adapter. Parsed graph indexes are cached by dataset and graph
database identity, with bounded LRU/TTL retention and explicit invalidation by
the code-graph write pipeline. Callers should select the narrowest dataset that
contains the repositories they want to traverse.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import heapq
import json
import math
import os
import re
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from threading import Lock, RLock
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from cognee.context_global_variables import current_dataset_id
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.config import get_graph_context_config
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.code_graph_diagram import DIAGRAM_FORMATS, render_result_diagram
from cognee.modules.retrieval.code_search_index import CodeSearchIndex, query_terms

CODE_NODE_TYPES = (
    "ApiEndpoint",
    "CodeAssociation",
    "CodeExtractionAccount",
    "CodeInsight",
    "CodeIntent",
    "CodeLintFinding",
    "CodeModule",
    "CodeService",
    "CodeSymbol",
    "CodeTestReference",
    "CodeFileReference",
    # Enola flows (flows.json) load with the code graph but are held aside
    # from the fact indexes; see _CodeGraphSnapshot.flows.
    "CodeFlow",
    "ExternalDependency",
    "StorageResource",
)

_KIND_BY_TYPE = {
    "ApiEndpoint": "route",
    "CodeAssociation": "association",
    "CodeExtractionAccount": "extraction",
    "CodeInsight": "insight",
    "CodeIntent": "intent",
    "CodeLintFinding": "lint",
    "CodeModule": "module",
    "CodeService": "service",
    "CodeSymbol": "symbol",
    "CodeTestReference": "test_ref",
    "CodeFileReference": "file_ref",
    "CodeFlow": "flow",
    "ExternalDependency": "dependency",
    "StorageResource": "storage",
}
_TYPE_BY_KIND = {kind: node_type for node_type, kind in _KIND_BY_TYPE.items()}
_FLOW_TYPE = "CodeFlow"
# The kinds node_types/kinds filters accept: flows are not facts, so no
# fact-shaped operation can select them.
_FACT_TYPE_BY_KIND = {
    kind: node_type for kind, node_type in _TYPE_BY_KIND.items() if node_type != _FLOW_TYPE
}
# The facts a framework extractor derives from a declaration and names after it
# (an ORM model's storage, a relation method's association), as node types;
# extract_code_graph._DECLARATION_VIEW_KINDS holds the same kinds.
_DECLARATION_VIEW_TYPES = frozenset({_TYPE_BY_KIND["storage"], _TYPE_BY_KIND["association"]})
_OPERATIONS = {
    "query_facts",
    "explore",
    "traverse",
    "find_path",
    "impact_analysis",
    "insights",
    "architecture",
    "delta",
    "capabilities",
    "search",
    "neighbors",
    "subgraph",
    "flows",
    "hubs",
    "ask",
}
# Results with no node/edge graph to draw: asking them for a diagram is an error.
_UNDRAWABLE_OPERATIONS = {"capabilities", "search", "flows", "hubs", "ask"}
# Edges that say where a fact lives rather than what it does. hubs and ask's
# neighbourhood skip them unless relation_types names them (has_method stays:
# a type's methods are its behaviour).
_CONTAINMENT_EDGE_TYPES = {"part_of", "declares", "evidences"}
_ROUTE_EDGE_TYPES = {"handled_by", "makes_request", "reaches_route"}
_SEARCH_DEFAULT_KINDS = ("symbol", "route", "flow", "module", "storage", "service")
# _match_tier of a fact's name against the whole query: an exact name (tier
# 3) or a short name / qualified suffix (tier 2) lifts the hit.
_EXACT_NAME_BONUS = {3: 5.0, 2: 2.5}
_TEST_PATH_SEGMENTS = {"test", "tests", "__tests__", "spec", "testdata"}
_TEST_FILE_NAME = re.compile(
    r"test_.*\.py|.*_test\.(?:py|go)|.*\.(?:test|spec)\.(?:js|jsx|ts|tsx)"
    r"|.*Test\.java|.*Tests\.cs|.*_spec\.rb"
)
# Enola walks a flow at most eight calls deep and stops at 5000 symbols; the
# reverse walk that finds the flows through a symbol keeps the same bounds.
_FLOW_WALK_MAX_DEPTH = 8
_FLOW_WALK_MAX_VISITED = 5000
_ASK_MAX_FLOWS = 3
_CAPABILITIES_VERSION = 1
# Defaults and caps of the navigation operations, reported verbatim by
# capabilities so a client can size its requests.
_LIMITS: dict[str, dict[str, Any]] = {
    "search": {"limit": {"default": 10, "max": 50}},
    "neighbors": {"limit": {"default": 50, "max": 200}},
    "subgraph": {
        "seeds": 20,
        "depth": {"default": 1, "max": 3},
        "max_nodes": {"default": 100, "max": 500},
        "max_edges": {"default": 300, "max": 1000},
    },
    "flows": {"limit": {"default": 20, "max": 50}},
    "hubs": {"limit": {"default": 20, "max": 100}},
    "ask": {
        "max_seeds": {"default": 5, "max": 10},
        "max_nodes": {"default": 60, "max": 200},
        "max_citations": {"default": 25, "max": 60},
    },
}
_NO_CODE_GRAPH_MESSAGE = (
    "CODE found no code graph in this dataset. Index a repository into it first (add or "
    "remember the repository, then cognify), or search the dataset that holds it."
)
_NO_FLOWS_MESSAGE = (
    "CODE flows are not available: this dataset's code graph has no flows. Flows come from "
    "Enola's flows.json, which an Enola release with flows support writes; re-index the "
    "repository with such a release to add them."
)
# The node kinds an architecture overview shows by default: the module
# structure plus what it exposes (routes), persists to (storage) and deploys
# as (services). Symbols, import edges and declared packages are rolled up
# into module-to-module edges rather than drawn.
_ARCHITECTURE_DEFAULT_TYPES = {"CodeModule", "ApiEndpoint", "StorageResource", "CodeService"}
# Edges that describe containment or synthesis rather than a dependency.
_ARCHITECTURE_SKIPPED_EDGE_TYPES = {"part_of", "has_method", "evidences"}
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
_CODE_GRAPH_SNAPSHOT_FORMAT_VERSION = 2
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
    """Invalid deterministic CODE operation or graph-node resolution.

    ``error_code`` is a stable, machine-readable reason (``unknown_operation``,
    ``invalid_argument``, ``not_found``, ``ambiguous``, ``no_code_graph``,
    ``feature_unavailable``) and ``error_param`` names the offending argument
    when there is one; the REST error handler returns both beside ``detail``.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_argument",
        param: str | None = None,
    ):
        super().__init__(message=message, name="CodeSearchValidationError", log=False)
        self.error_code = code
        self.error_param = param


class CodeSeedNotFoundError(CodeSearchValidationError):
    """A seed name/id did not resolve (or was ambiguous) in this dataset's graph.

    Distinct from argument-validation errors: in a multi-dataset search, a seed
    that one dataset cannot resolve is a per-dataset miss, not a bad request,
    so the search layer reports it per dataset instead of failing the call.
    An ambiguous name carries ``error_code="ambiguous"``.
    """

    def __init__(self, message: str, *, code: str = "not_found", param: str | None = None):
        super().__init__(message, code=code, param=param)


class CodeGraphUnavailableError(CodeSeedNotFoundError):
    """The dataset has no code graph (``no_code_graph``), or its graph lacks the
    data an operation needs (``feature_unavailable``: flows without flows.json).

    It derives from the seed miss on purpose: both say "this dataset cannot
    answer", not "the request is wrong". A multi-dataset search reports either
    one per dataset (search.py ``_report_code_seed_miss``) instead of failing
    the call, and recall's code lane turns either into "no code facts".
    """

    def __init__(self, message: str, *, code: str = "no_code_graph", param: str | None = None):
        super().__init__(message, code=code, param=param)


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
        raise CodeSearchValidationError(
            f"CODE {field} must be a string or a list of strings.", param=field
        )
    if not all(isinstance(item, str) and item for item in value):
        raise CodeSearchValidationError(
            f"CODE {field} must contain only non-empty strings.", param=field
        )
    return sorted(set(value)) if isinstance(value, set) else list(dict.fromkeys(value))


def _bounded_int(value: Any, *, field: str, default: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodeSearchValidationError(f"CODE {field} must be an integer.", param=field)
    if value <= 0:
        return default
    return min(value, maximum)


def _count(value: Any, *, field: str, default: int, maximum: int) -> int:
    """A limit of the navigation operations: like _bounded_int, but negative is an error.

    None and 0 mean the default; values above the maximum are capped.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value < 0:
        raise CodeSearchValidationError(
            f"CODE {field} must be a non-negative integer.", param=field
        )
    return _bounded_int(value, field=field, default=default, maximum=maximum)


def _offset(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CodeSearchValidationError(
            "CODE offset must be a non-negative integer.", param="offset"
        )
    return value


def _confidence(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise CodeSearchValidationError(
            f"CODE {field} must be a number between 0 and 1.", param=field
        )
    return float(value)


def _optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise CodeSearchValidationError(f"CODE {field} must be a boolean.", param=field)


def _direction(value: Any, *, default: str) -> str:
    direction = str(value or default).lower()
    if direction not in {"forward", "reverse", "both"}:
        raise CodeSearchValidationError(
            "CODE direction must be 'forward', 'reverse', or 'both'.", param="direction"
        )
    return direction


def _node_type(value: str, field: str = "node_types") -> str:
    if value in _FACT_TYPE_BY_KIND.values():
        return value
    normalized = value.strip().lower()
    if normalized in _FACT_TYPE_BY_KIND:
        return _FACT_TYPE_BY_KIND[normalized]
    for candidate in _FACT_TYPE_BY_KIND.values():
        if candidate.lower() == normalized:
            return candidate
    raise CodeSearchValidationError(
        f"Unsupported CODE node kind/type {value!r}; expected one of "
        f"{', '.join(sorted(_FACT_TYPE_BY_KIND))}.",
        param=field,
    )


def _node_types(value: Any, field: str = "node_types") -> set[str] | None:
    values = _as_strings(value, field)
    return {_node_type(item, field) for item in values} if values else None


def _is_test_path(file_path: str) -> bool:
    """Whether a path looks like test code: a test directory or a test file name."""
    parts = [part for part in file_path.replace("\\", "/").split("/") if part]
    if not parts:
        return False
    if any(part.lower() in _TEST_PATH_SEGMENTS for part in parts):
        return True
    return _TEST_FILE_NAME.fullmatch(parts[-1]) is not None


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    """The mapping items of a list prop (Enola refs, stages); anything else is empty."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


@dataclass(frozen=True)
class _CodeTextIndex:
    """The text-search structures, built on first use: BM25 plus a name lookup."""

    bm25: CodeSearchIndex
    # Casefolded full names and _short_names forms -> node/flow ids, so an
    # exact or short-name match is found even when every query token is a
    # stopword.
    name_forms: dict[str, tuple[str, ...]]


class _CodeGraphSnapshot:
    """Stable in-memory indexes and adjacency over one dataset-scoped graph."""

    def __init__(self, raw_nodes: Iterable[Any], raw_edges: Iterable[Any]):
        self.nodes: dict[str, dict[str, Any]] = {}
        # Repository nodes are deliberately NOT facts (kept out of every fact
        # index and traversal); they are held aside only for the delta
        # operation, which reports the last ingestion's changes.
        self.repositories: dict[str, dict[str, Any]] = {}
        # Flow nodes (Enola's flows.json) are held aside the same way, so no
        # fact operation sees them; flows, search, ask and capabilities read
        # them through the flow indexes built below.
        self.flows: dict[str, dict[str, Any]] = {}
        for item in raw_nodes or []:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            raw_id, raw_properties = item
            node_id = str(raw_id)
            properties = _normalize_properties(raw_properties)
            # Every node written since docs were read has a doc field, null
            # when undocumented. Ladybug keeps the null; Neo4j and older nodes
            # have no key. Drop it so ``doc`` is present exactly on documented
            # facts and the property filters match on documented facts only.
            doc = properties.get("doc", _MISSING)
            if doc is not _MISSING and not (isinstance(doc, str) and doc.strip()):
                del properties["doc"]
            node_type = properties.get("type")
            if node_type == "CodeRepository":
                properties["id"] = node_id
                self.repositories[node_id] = properties
                continue
            if node_type == _FLOW_TYPE:
                properties["id"] = node_id
                self.flows[node_id] = properties
                continue
            if node_type not in CODE_NODE_TYPES:
                continue
            properties["id"] = node_id
            self.nodes[node_id] = properties

        normalized_edges = []
        flow_edges: set[tuple[str, str, str]] = set()
        for item in raw_edges or []:
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                continue
            source_id, target_id, relationship = map(str, item[:3])
            if source_id in self.flows and target_id in self.nodes:
                flow_edges.add((source_id, relationship, target_id))
                continue
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
        # enola's own fact id (32 hex chars) -> cognee node id, so callers can
        # seed operations with ids copied from facts.jsonl or from enola's MCP
        # tools. Several enola ids may collapse onto one node; the node keeps
        # the first, so only that one is addressable here.
        self.by_enola_id: dict[str, str] = {}
        self.by_property: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for node_id, node in sorted(
            self.nodes.items(), key=lambda item: self.node_sort_key(item[0])
        ):
            self.by_type[str(node.get("type", ""))].add(node_id)
            self.by_name[str(node.get("name", ""))].add(node_id)
            self.by_file[self.file_of(node)].add(node_id)
            self.by_repo[str(node.get("repo", ""))].add(node_id)
            enola_id = node.get("enola_id")
            if isinstance(enola_id, str) and enola_id:
                self.by_enola_id.setdefault(enola_id, node_id)
            for key, value in node.items():
                if key != "fact_properties":
                    self._index_property(node_id, key, value)
            for key, value in self.properties_of(node).items():
                self._index_property(node_id, key, value)

        self.declaration_of = self._index_declarations()
        self.documented = {node_id for node_id, node in self.nodes.items() if self.doc_of(node)}
        # Test code (see _is_test_path), judged once per file: search, ask and
        # hubs leave it out unless include_tests is set.
        self.test_nodes = set(self.by_type.get("CodeTestReference", ()))
        for file_path, node_ids in self.by_file.items():
            if _is_test_path(file_path):
                self.test_nodes |= node_ids
        self._index_flows(flow_edges)
        self._text_index: _CodeTextIndex | None = None
        self._text_index_lock = Lock()

    def _index_declarations(self) -> dict[str, str]:
        """Storage and association node ids -> the CodeSymbol each was derived from.

        A framework extractor names such a view after its declaration (an ORM
        model, a relation method) and places it in the same file. A view whose
        name, repo and file match no single symbol maps to nothing.
        """
        declarations = {}
        for node_type in _DECLARATION_VIEW_TYPES:
            for view_id in self.by_type.get(node_type, ()):
                view = self.nodes[view_id]
                file_path = self.file_of(view)
                symbols = [
                    node_id
                    for node_id in self.by_name.get(str(view.get("name", "")), ())
                    if self.nodes[node_id].get("type") == "CodeSymbol"
                    and self.nodes[node_id].get("repo") == view.get("repo")
                    and self.file_of(self.nodes[node_id]) == file_path
                ]
                if file_path and len(symbols) == 1:
                    declarations[view_id] = symbols[0]
        return declarations

    def _index_property(self, node_id: str, name: str, value: Any) -> None:
        self.by_property[name][_scalar_text(value)].add(node_id)
        if isinstance(value, Mapping):
            for child_name, child_value in value.items():
                self._index_property(node_id, f"{name}.{child_name}", child_value)

    def _index_flows(self, flow_edges: set[tuple[str, str, str]]) -> None:
        """Resolve each flow's entry, stages and routes to fact nodes, both ways.

        ``flow_members`` maps a flow to the node ids of its entry, stages and
        routes (None where a ref names no node), aligned with the refs in its
        properties; ``flows_by_node`` maps a node to the (flow id, role) pairs
        it takes part in.
        """
        edge_targets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for source_id, relationship, target_id in sorted(flow_edges):
            edge_targets[source_id][relationship].append(target_id)
        self.flow_by_enola_id: dict[str, str] = {}
        self.flow_members: dict[str, dict[str, Any]] = {}
        self.flows_by_node: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.flow_order = sorted(self.flows, key=self.flow_rank_key)
        for flow_id in self.flow_order:
            flow = self.flows[flow_id]
            enola_id = flow.get("enola_id")
            if isinstance(enola_id, str) and enola_id:
                self.flow_by_enola_id.setdefault(enola_id, flow_id)
            properties = self.properties_of(flow)
            targets = edge_targets.get(flow_id, {})
            entry = self._flow_ref_node(properties.get("entry"), targets.get("flow_entry", ()))
            stages = [
                self._flow_ref_node(stage, targets.get("flow_stage", ()))
                for stage in _mappings(properties.get("stages"))
            ]
            routes = [
                self._flow_ref_node(route, targets.get("flow_route", ()))
                for route in _mappings(properties.get("routes"))
            ]
            self.flow_members[flow_id] = {"entry": entry, "stages": stages, "routes": routes}
            for role, node_ids in (("entry", [entry]), ("stage", stages), ("route", routes)):
                for node_id in dict.fromkeys(node_ids):
                    if node_id is not None:
                        self.flows_by_node[node_id].append((flow_id, role))

    def _flow_ref_node(self, ref: Any, edge_targets: Iterable[str]) -> str | None:
        """The fact node an Enola ref names: by its fact id, else by the flow's edge."""
        if not isinstance(ref, Mapping):
            return None
        enola_id = ref.get("id")
        if isinstance(enola_id, str) and enola_id in self.by_enola_id:
            return self.by_enola_id[enola_id]
        # A node keeps only the first of the enola ids that collapse onto it;
        # the flow's own edges were resolved through every id at ingestion.
        for node_id in edge_targets:
            if self.nodes[node_id].get("name") == ref.get("name"):
                return node_id
        return None

    @staticmethod
    def file_of(node: Mapping[str, Any]) -> str:
        return str(node.get("file_path") or node.get("file") or "")

    @staticmethod
    def properties_of(node: Mapping[str, Any]) -> dict[str, Any]:
        properties = node.get("fact_properties")
        return properties if isinstance(properties, dict) else {}

    @classmethod
    def doc_of(cls, node: Mapping[str, Any]) -> str | None:
        """A fact's docstring: its ``doc`` field, else the ``doc`` prop older loads kept.

        Read by ingestion's rule: an insight has none (its explainer prose is
        its description), and blank text is no doc.
        """
        if node.get("type") == "CodeInsight":
            return None
        for value in (node.get("doc"), cls.properties_of(node).get("doc")):
            if isinstance(value, str) and value.strip():
                return value
        return None

    def entity(self, key: str) -> dict[str, Any]:
        """A fact node or a flow node, by cognee id."""
        return self.nodes[key] if key in self.nodes else self.flows[key]

    def is_test(self, key: str) -> bool:
        if key in self.nodes:
            return key in self.test_nodes
        return _is_test_path(self.file_of(self.flows[key]))

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

    def entity_sort_key(self, key: str) -> tuple[str, ...]:
        """node_sort_key for facts and flows alike."""
        if key in self.nodes:
            return self.node_sort_key(key)
        flow = self.flows[key]
        return (
            str(flow.get("repo", "")),
            _FLOW_TYPE,
            str(flow.get("name", "")).casefold(),
            self.file_of(flow),
            str(flow.get("line", "")),
            key,
        )

    def flow_rank_key(self, flow_id: str) -> tuple[Any, ...]:
        """Enola's ranking: rank, then name (repo and id keep ties stable)."""
        flow = self.flows[flow_id]
        rank = _int_or_none(self.properties_of(flow).get("rank"))
        return (
            math.inf if rank is None else rank,
            str(flow.get("name") or "").casefold(),
            str(flow.get("repo") or ""),
            flow_id,
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
        depth: int | None = None,
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
            "end_line": node.get("end_line"),
            "repo": node.get("repo"),
            "path": node.get("path"),
            "description": node.get("description"),
            "symbol_kind": node.get("symbol_kind"),
            "enola_id": node.get("enola_id"),
        }
        result.update({key: value for key, value in aliases.items() if value not in (None, "")})
        doc = self.doc_of(node)
        if doc:
            result["doc"] = doc
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
        name: str | None = None,
        node_id: str | None = None,
        repo: str | None = None,
        role: str = "node",
    ) -> str:
        if node_id:
            normalized_id = str(node_id)
            if normalized_id not in self.nodes and normalized_id in self.by_enola_id:
                # An enola fact id (from facts.jsonl or enola's own tools).
                normalized_id = self.by_enola_id[normalized_id]
            if normalized_id not in self.nodes:
                raise CodeSeedNotFoundError(
                    f"CODE could not resolve {role} id {normalized_id!r}.", param="id"
                )
            if repo and self.nodes[normalized_id].get("repo") != repo:
                raise CodeSeedNotFoundError(
                    f"CODE {role} id {normalized_id!r} is not in repo {repo!r}.", param="id"
                )
            return normalized_id

        term = str(name or "").strip()
        if not term:
            raise CodeSearchValidationError(
                f"CODE {role} requires a non-empty name or id.", param="name"
            )
        best = self.name_matches(term, repo=repo)
        if not best:
            raise CodeSeedNotFoundError(
                f"CODE could not resolve {role} name {term!r}.", param="name"
            )
        if len(best) > 1:
            # One declaration seen twice (a class or method symbol plus the
            # storage or association view a framework extractor names after
            # it) is one candidate: the symbol.
            matched = set(best)
            rivals = [item for item in best if self.declaration_of.get(item) not in matched]
            if len(rivals) == 1:
                return rivals[0]
            choices = ", ".join(
                f"{self.nodes[item].get('name')} [{self.nodes[item].get('type')}, "
                f"repo={self.nodes[item].get('repo') or '-'}, id={item}]"
                for item in best[:10]
            )
            raise CodeSeedNotFoundError(
                f"CODE {role} name {term!r} is ambiguous; use an exact id or repo. "
                f"Candidates: {choices}",
                code="ambiguous",
                param="name",
            )
        return best[0]

    def name_matches(self, term: str, *, repo: str | None = None) -> list[str]:
        """The nodes whose name matches term best: exact, else short name, else substring."""
        lowered = term.strip().casefold()
        ranked: list[tuple[int, str]] = []
        for candidate_id, node in self.nodes.items():
            candidate_name = str(node.get("name") or "")
            if repo and node.get("repo") != repo:
                continue
            tier = _match_tier(candidate_name, lowered)
            if tier >= 0:
                ranked.append((tier, candidate_id))
        if not ranked:
            return []
        best_tier = max(tier for tier, _ in ranked)
        return sorted(
            (candidate_id for tier, candidate_id in ranked if tier == best_tier),
            key=self.node_sort_key,
        )

    def text_index(self) -> _CodeTextIndex:
        """The search index, built on first use and kept for the snapshot's lifetime."""
        index = self._text_index
        if index is None:
            with self._text_index_lock:
                index = self._text_index
                if index is None:
                    index = self._text_index = self._build_text_index()
        return index

    def _build_text_index(self) -> _CodeTextIndex:
        keys = sorted(self.nodes, key=self.node_sort_key)
        keys.extend(sorted(self.flows, key=self.entity_sort_key))
        documents = []
        name_forms: dict[str, set[str]] = defaultdict(set)
        for key in keys:
            entity = self.entity(key)
            name = str(entity.get("name") or "")
            fields = {"name": name, "doc": self.doc_of(entity), "file": self.file_of(entity)}
            if key in self.flows:
                fields["summary"] = self.flow_summary_text(key)
            documents.append((key, fields))
            if name:
                name_forms[name.casefold()].add(key)
                for form in _short_names(name):
                    name_forms[form].add(key)
        return _CodeTextIndex(
            bm25=CodeSearchIndex(documents),
            name_forms={form: tuple(sorted(found)) for form, found in name_forms.items()},
        )

    def calls_reach(self, node_id: str) -> tuple[dict[str, int], bool]:
        """Callers of node_id, transitively: node id -> calls hops, and whether the cap cut it.

        A reverse walk over ``calls`` edges, bounded like Enola's own flow walk
        (eight hops, 5000 symbols). It reads the adjacency lists directly.
        """
        distances = {node_id: 0}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            depth = distances[current]
            if depth >= _FLOW_WALK_MAX_DEPTH:
                continue
            for edge in self.reverse.get(current, ()):
                if edge["type"] != "calls" or edge["source_id"] in distances:
                    continue
                if len(distances) >= _FLOW_WALK_MAX_VISITED:
                    return distances, True
                distances[edge["source_id"]] = depth + 1
                queue.append(edge["source_id"])
        return distances, False

    def flow_summary_text(self, flow_id: str) -> str | None:
        flow = self.flows[flow_id]
        summary = self.properties_of(flow).get("summary") or flow.get("description")
        return summary if isinstance(summary, str) else None

    def flow_ref(self, ref: Any, node_id: str | None) -> dict[str, Any]:
        """An Enola ref (entry, stage, route) with the cognee node it resolves to.

        ``node_id`` is None when no fact node carries the ref's id or name. The
        ref's own file, line and doc come first; the node fills what it lacks.
        """
        ref = ref if isinstance(ref, Mapping) else {}
        node = self.nodes.get(node_id) if node_id else None
        result: dict[str, Any] = {
            "name": ref.get("name") or (node.get("name") if node else None),
            "enola_id": ref.get("id"),
            "node_id": node_id,
        }
        optional = {
            "file": ref.get("file") or (self.file_of(node) if node else None),
            "line": ref.get("line") if ref.get("line") is not None else (node or {}).get("line"),
            "doc": ref.get("doc") or (self.doc_of(node) if node else None),
            "method": ref.get("method"),
        }
        result.update({key: value for key, value in optional.items() if value not in (None, "")})
        return result

    def flow_summary(self, flow_id: str) -> dict[str, Any]:
        """One flow as a list item: identity, ranking, entry, routes, summary, metrics."""
        flow = self.flows[flow_id]
        properties = self.properties_of(flow)
        members = self.flow_members[flow_id]
        result: dict[str, Any] = {
            "id": flow_id,
            "enola_id": flow.get("enola_id"),
            "name": flow.get("name"),
            "rank": properties.get("rank"),
            "repo": flow.get("repo"),
            "mechanism": properties.get("mechanism"),
        }
        if properties.get("side") is True:
            result["side"] = True
        result["entry"] = self.flow_ref(properties.get("entry"), members["entry"])
        routes = [
            self.flow_ref(route, node_id)
            for route, node_id in zip(_mappings(properties.get("routes")), members["routes"])
        ]
        if routes:
            result["routes"] = routes
        result["summary"] = self.flow_summary_text(flow_id)
        metrics = properties.get("metrics")
        result["metrics"] = _normalize_value(metrics) if isinstance(metrics, Mapping) else {}
        return result

    def flow_detail(self, flow_id: str) -> dict[str, Any]:
        """flow_summary plus the stages, modules and the flows it overlaps."""
        result = self.flow_summary(flow_id)
        properties = self.properties_of(self.flows[flow_id])
        stages = []
        for stage, node_id in zip(
            _mappings(properties.get("stages")), self.flow_members[flow_id]["stages"]
        ):
            item = self.flow_ref(stage, node_id)
            for key in ("order", "depth", "parent", "reach"):
                if stage.get(key) is not None:
                    item[key] = stage[key]
            stages.append(item)
        result["stages"] = stages
        result["modules"] = _normalize_value(_mappings(properties.get("modules")))
        for key in ("same_reach_as", "nested_in"):
            if properties.get(key):
                result[key] = _normalize_value(properties[key])
        return result

    def flows_reaching(
        self, node_id: str, reached: Mapping[str, int]
    ) -> dict[str, tuple[str, int]]:
        """Flow id -> (via, distance) for every flow that goes through node_id.

        ``reached`` is calls_reach(node_id). A flow goes through the node when
        its entry is the node or calls down to it (via "entry", "stage" when
        Enola listed the node as a stage, else "reach"; distance in calls
        hops); a stage or route the walk missed still counts, at the stage's
        depth or 0.
        """
        found: dict[str, tuple[str, int]] = {}
        for flow_id in self.flow_order:
            members = self.flow_members[flow_id]
            entry = members["entry"]
            if entry is not None and entry in reached:
                if entry == node_id:
                    via = "entry"
                elif node_id in members["stages"]:
                    via = "stage"
                else:
                    via = "reach"
                found[flow_id] = (via, reached[entry])
            elif node_id in members["stages"]:
                stages = _mappings(self.properties_of(self.flows[flow_id]).get("stages"))
                position = members["stages"].index(node_id)
                depth = (
                    _int_or_none(stages[position].get("depth")) if position < len(stages) else None
                )
                found[flow_id] = ("stage", depth or 1)
            elif node_id in members["routes"]:
                found[flow_id] = ("route", 0)
        return found


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
    graph_config: Mapping[str, Any] | None = None,
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
    graph_config: Mapping[str, Any] | None = None,
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
    # Every tier below needs the term inside the name (each short-name form
    # is a part of it); most names fail here, before _short_names runs.
    if lowered_term not in lowered_name:
        return -1
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
        config: Mapping[str, Any] | None = None,
        retriever_specific_config: Mapping[str, Any] | None = None,
        **operation_config: Any,
    ):
        combined = dict(retriever_specific_config or {})
        combined.update(dict(config or {}))
        combined.update(operation_config)
        operation = str(combined.get("operation") or "explore").lower()
        if operation not in _OPERATIONS:
            raise CodeSearchValidationError(
                f"Unsupported CODE operation {operation!r}; expected one of "
                f"{', '.join(sorted(_OPERATIONS))}.",
                code="unknown_operation",
                param="operation",
            )
        self.operation = operation
        self.config = combined
        self.diagram_format = self._diagram_format(combined.get("diagram"), operation)

    @staticmethod
    def _diagram_format(value: Any, operation: str) -> str | None:
        """Which diagram to attach to the result, if any.

        ``diagram`` may be a format name ("mermaid", "dot"), True (Mermaid) or
        False/None (none). The architecture overview draws itself by default,
        since a picture is the point of it; every other operation is opt-in.
        Operations whose result has no graph (search, ask, flows, hubs,
        capabilities) reject a diagram request.
        """
        if value is None:
            return "mermaid" if operation == "architecture" else None
        if value is False:
            return None
        if value is True:
            diagram_format = "mermaid"
        elif isinstance(value, str) and value.strip().lower() in DIAGRAM_FORMATS:
            diagram_format = value.strip().lower()
        else:
            raise CodeSearchValidationError(
                f"Unsupported CODE diagram {value!r}; expected one of "
                f"{', '.join(DIAGRAM_FORMATS)}, or a boolean.",
                param="diagram",
            )
        if operation in _UNDRAWABLE_OPERATIONS:
            raise CodeSearchValidationError(
                f"CODE {operation} results carry no graph to draw; drop diagram, or use "
                "subgraph or neighbors for a picture.",
                param="diagram",
            )
        return diagram_format

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
                [{"type": [*CODE_NODE_TYPES, "CodeRepository"]}]
            )
            return _CodeGraphSnapshot(nodes, edges)

        return await _CODE_GRAPH_SNAPSHOT_CACHE.get_or_load(key, load)

    async def get_retrieved_objects(self, query: str | None, query_batch=None) -> dict[str, Any]:
        if query_batch is not None:
            raise CodeSearchValidationError(
                "SearchType.CODE does not support batched queries.", param="query_batch"
            )
        snapshot = await self._snapshot()
        handler = getattr(self, f"_{self.operation}")
        result = handler(snapshot, query or "")
        if self.diagram_format is not None:
            result["diagram"] = render_result_diagram(result, self.diagram_format)
        return result

    async def get_context_from_objects(
        self,
        query: str | None = None,
        query_batch=None,
        retrieved_objects: Any = None,
    ) -> str:
        return _json_key(retrieved_objects)

    async def get_completion_from_context(
        self,
        query: str | None = None,
        query_batch=None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        return retrieved_objects

    def _delta(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """What the last ingestion changed, per repository.

        Reads the last_delta record that add_code_graph_edges stamps on each
        CodeRepository node after a completed load + sweep (counts of added/
        updated/removed nodes and edges, capped name samples, the snapshot id
        and load timestamp). A repository loaded before delta stamping existed
        reports delta: None.
        """
        repo_filter = self.config.get("repo") or self.config.get("name") or query or ""
        repo_filter = str(repo_filter).strip().casefold()
        repositories = []
        for node_id in sorted(graph.repositories):
            node = graph.repositories[node_id]
            repo_name = str(node.get("name") or "")
            if repo_filter and repo_filter not in repo_name.casefold():
                continue
            repositories.append(
                {
                    "id": node_id,
                    "repo": repo_name,
                    "last_snapshot_id": node.get("last_snapshot_id"),
                    "delta": node.get("last_delta"),
                    # receipt.json projection stamped with the snapshot: format
                    # and enola versions, git provenance, counts and the
                    # extraction-quality block (parse errors, files parsed vs
                    # seen, skip census). None before receipt stamping existed.
                    "receipt": _normalize_value(node.get("last_receipt")),
                }
            )
        return {"operation": "delta", "repositories": repositories}

    def _architecture(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """Module-level overview of the code graph, drawn as a diagram by default.

        Shows modules plus the routes, storage and services they declare
        (``node_types`` overrides the set), and rolls every other edge up to
        the modules that declare its endpoints: two symbols calling across
        modules become one ``calls`` edge between the modules, carrying a
        ``count``. Edges internal to one module are dropped. ``repo`` narrows
        to one repository; ``max_nodes`` (default 80) keeps the picture
        readable by keeping the best-connected nodes and reporting
        ``truncated``.
        """
        repo = self.config.get("repo")
        node_types = _node_types(self.config.get("node_types")) or set(_ARCHITECTURE_DEFAULT_TYPES)
        max_nodes = _bounded_int(
            self.config.get("max_nodes"), field="max_nodes", default=80, maximum=300
        )
        relation_types = self._relation_types()

        def in_scope(node_id: str) -> bool:
            node = graph.nodes[node_id]
            return node.get("type") in node_types and (repo is None or node.get("repo") == repo)

        module_cache: dict[str, str | None] = {}

        def module_of(node_id: str) -> str | None:
            """The module that declares a fact: its `declares` edge, else its file's directory."""
            if node_id in module_cache:
                return module_cache[node_id]
            node = graph.nodes[node_id]
            found: str | None = None
            if node.get("type") == "CodeModule":
                found = node_id
            else:
                for edge in graph.forward.get(node_id, []):
                    if (
                        edge["type"] == "declares"
                        and graph.nodes[edge["target_id"]].get("type") == "CodeModule"
                    ):
                        found = edge["target_id"]
                        break
                if found is None:
                    directory = (
                        graph.file_of(node).rsplit("/", 1)[0] if "/" in graph.file_of(node) else "."
                    )
                    while found is None:
                        candidates = sorted(
                            candidate
                            for candidate in graph.by_name.get(directory, set())
                            if graph.nodes[candidate].get("type") == "CodeModule"
                            and graph.nodes[candidate].get("repo") == node.get("repo")
                        )
                        if candidates:
                            found = candidates[0]
                        elif directory in (".", ""):
                            break
                        else:
                            directory = directory.rsplit("/", 1)[0] if "/" in directory else "."
            module_cache[node_id] = found
            return found

        def endpoint(node_id: str) -> str | None:
            return node_id if in_scope(node_id) else module_of(node_id)

        rolled: dict[tuple[str, str, str], int] = defaultdict(int)
        for edge in graph.edges:
            if edge["type"] in _ARCHITECTURE_SKIPPED_EDGE_TYPES:
                continue
            if relation_types is not None and edge["type"] not in relation_types:
                continue
            source = endpoint(edge["source_id"])
            target = endpoint(edge["target_id"])
            if source is None or target is None or source == target:
                continue
            if not in_scope(source) or not in_scope(target):
                continue
            if edge["type"] == "declares" and graph.nodes[source].get("type") != "CodeModule":
                # A route/storage fact declares its module; draw it the way an
                # architecture reads — the module declares (exposes) the fact.
                source, target = target, source
            rolled[(source, target, edge["type"])] += 1

        degree: dict[str, int] = defaultdict(int)
        for source, target, _type in rolled:
            degree[source] += 1
            degree[target] += 1
        candidates = [node_id for node_id in graph.nodes if in_scope(node_id)]
        candidates.sort(key=lambda node_id: (-degree[node_id], graph.node_sort_key(node_id)))
        kept = candidates[:max_nodes]
        kept_set = set(kept)
        kept.sort(key=graph.node_sort_key)

        edges = []
        for (source, target, edge_type), count in sorted(
            rolled.items(),
            key=lambda item: (
                item[0][2],
                graph.node_sort_key(item[0][0]),
                graph.node_sort_key(item[0][1]),
            ),
        ):
            if source in kept_set and target in kept_set:
                edges.append(
                    {
                        "source_id": source,
                        "source": graph.nodes[source].get("name"),
                        "target_id": target,
                        "target": graph.nodes[target].get("name"),
                        "type": edge_type,
                        "count": count,
                    }
                )
        repos = sorted({str(graph.nodes[node_id].get("repo") or "") for node_id in kept})
        return {
            "operation": "architecture",
            "repos": [item for item in repos if item],
            "nodes": [graph.fact(node_id, include_relations=False) for node_id in kept],
            "edges": edges,
            "stats": {
                "nodes_total": len(candidates),
                "nodes_shown": len(kept),
                "edges_shown": len(edges),
                "truncated": len(candidates) > len(kept),
            },
        }

    def _insights(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """Explainer findings (CodeInsight nodes) with the facts they cite.

        Filters: ``source``/``sources`` (explainer names such as cycles,
        hotspots, god-class), ``min_confidence`` (0-1; enola scores
        structural findings 1.0 and heuristic ones below), ``informational``
        (True keeps only describe-the-graph findings, False drops them),
        ``repo``, and a title substring via ``name`` (or the query text when
        no other filter is set). Each insight carries ``evidence``: the facts
        its ``evidences`` edges resolve to. Ordered by confidence descending,
        then the stable node order; paginated like query_facts.
        """
        sources = set(_as_strings(self.config.get("sources"), "sources"))
        if self.config.get("source"):
            sources.add(str(self.config["source"]))
        min_confidence = _confidence(self.config.get("min_confidence"), field="min_confidence")
        informational = _optional_bool(self.config.get("informational"), field="informational")
        repo = self.config.get("repo")
        query_name = query if set(self.config).issubset({"operation"}) else ""
        substring = str(self.config.get("name") or query_name).strip().casefold()

        def confidence_of(node_id: str) -> float | None:
            value = graph.properties_of(graph.nodes[node_id]).get("confidence")
            return (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None
            )

        matched = []
        for node_id in graph.by_type.get("CodeInsight", set()):
            node = graph.nodes[node_id]
            properties = graph.properties_of(node)
            if sources and properties.get("source") not in sources:
                continue
            if min_confidence is not None:
                confidence = confidence_of(node_id)
                if confidence is None or confidence < min_confidence:
                    continue
            if informational is not None and bool(properties.get("informational")) != informational:
                continue
            if repo is not None and node.get("repo") != repo:
                continue
            if substring and substring not in str(node.get("name") or "").casefold():
                continue
            matched.append(node_id)

        matched.sort(
            key=lambda node_id: (
                -(confidence_of(node_id) if confidence_of(node_id) is not None else -1.0),
                graph.node_sort_key(node_id),
            )
        )
        by_source: dict[str, int] = defaultdict(int)
        for node_id in matched:
            by_source[
                str(graph.properties_of(graph.nodes[node_id]).get("source") or "unknown")
            ] += 1

        total = len(matched)
        offset = _offset(self.config.get("offset"))
        limit = _bounded_int(self.config.get("limit"), field="limit", default=50, maximum=500)
        page = matched[offset : offset + limit]
        insights = []
        for node_id in page:
            insight = graph.fact(node_id, include_relations=False)
            insight["evidence"] = [
                graph.fact(edge["target_id"], include_relations=False)
                for edge in graph.forward.get(node_id, [])
                if edge["type"] == "evidences"
            ]
            insights.append(insight)
        return {
            "operation": "insights",
            "insights": insights,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < total,
            "by_source": dict(sorted(by_source.items())),
        }

    def _query_facts(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        kinds = _as_strings(self.config.get("kinds"), "kinds")
        if self.config.get("kind"):
            kinds.append(str(self.config["kind"]))
        type_filter = {_node_type(kind, "kinds") for kind in kinds} if kinds else None

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
        if files and not file_prefix:
            # With file_prefix present the file condition is a union (exact
            # file OR prefix), which the exact-file index cannot express;
            # leave it to the post-filter below.
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
            if (
                (files or file_prefix)
                and file_name not in files
                and not (file_prefix and file_name.startswith(file_prefix))
            ):
                continue
            node_name = str(node.get("name") or "")
            if (
                (names or substring)
                and node_name not in names
                and not (substring and substring in node_name.casefold())
            ):
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
        # A fact_properties key may be shadowed by an always-present top-level
        # field (type, name, line, ...); both sources are indexed, so both
        # must be able to satisfy the filter.
        for source in (node, _CodeGraphSnapshot.properties_of(node)):
            value: Any = source
            found = True
            for part in property_name.split("."):
                if isinstance(value, Mapping) and part in value:
                    value = value[part]
                else:
                    found = False
                    break
            if found and (expected is _MISSING or _scalar_text(value) == _scalar_text(expected)):
                return True
        return False

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
        matched_target_id: str | None = None
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
            raise CodeSearchValidationError(
                f"CODE {role} requires query_text, names, or node_ids.", param="names"
            )
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

    # --- Agent navigation: capabilities, search, neighbors, subgraph, flows, hubs, ask ---

    def _limit(self, operation: str, field: str) -> int:
        bounds = _LIMITS[operation][field]
        return _count(
            self.config.get(field), field=field, default=bounds["default"], maximum=bounds["max"]
        )

    def _text_arg(self, query: str, *fields: str) -> str:
        """The first non-empty string among config fields, else the query text."""
        for field in fields:
            value = self.config.get(field)
            if value is None:
                continue
            if not isinstance(value, str):
                raise CodeSearchValidationError(f"CODE {field} must be a string.", param=field)
            if value.strip():
                return value.strip()
        return (query or "").strip()

    def _identifier_arg(self, *fields: str) -> str | None:
        """The first non-empty id, name or repo among config fields, as a string.

        A list, number or other non-string is a usage error (``invalid_argument``),
        not an identifier that fails to resolve: ``not_found`` is softened into a
        per-dataset miss, which would hide the caller's mistake.
        """
        for field in fields:
            value = self.config.get(field)
            if value is None:
                continue
            if isinstance(value, UUID):
                return str(value)
            if not isinstance(value, str):
                raise CodeSearchValidationError(
                    f"CODE {field} must be a string, got {type(value).__name__}.", param=field
                )
            if value.strip():
                return value.strip()
        return None

    def _include_tests(self) -> bool:
        return bool(_optional_bool(self.config.get("include_tests"), field="include_tests"))

    def _search_kinds(self) -> set[str] | None:
        """Node types a text search covers, flows included; None means every kind."""
        kinds = _as_strings(self.config.get("kinds"), "kinds") or list(_SEARCH_DEFAULT_KINDS)
        if any(kind.strip().lower() == "all" for kind in kinds):
            return None
        types = set()
        for kind in kinds:
            if kind.strip().lower() in {"flow", "codeflow"}:
                types.add(_FLOW_TYPE)
                continue
            try:
                types.add(_node_type(kind, "kinds"))
            except CodeSearchValidationError:
                # The fact kinds' own message leaves out what only a text
                # search takes.
                raise CodeSearchValidationError(
                    f"Unsupported CODE node kind/type {kind!r}; expected one of "
                    f"{', '.join(sorted([*_FACT_TYPE_BY_KIND, 'flow']))}, or all.",
                    param="kinds",
                ) from None
        return types

    @staticmethod
    def _incident_edges(
        graph: _CodeGraphSnapshot, node_id: str, direction: str
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """(neighbour, edge) pairs read straight off the sorted adjacency lists."""
        if direction in {"forward", "both"}:
            for edge in graph.forward.get(node_id, ()):
                yield edge["target_id"], edge
        if direction in {"reverse", "both"}:
            for edge in graph.reverse.get(node_id, ()):
                yield edge["source_id"], edge

    @staticmethod
    def _text_matches(
        graph: _CodeGraphSnapshot,
        text: str,
        *,
        kinds: set[str] | None,
        repo: str | None,
        file_prefix: str,
        include_tests: bool,
        limit: int | None = None,
    ) -> tuple[list[str], list[tuple[str, float, tuple[str, ...]]]]:
        """Rank facts and flows against text: (query terms, [(id, score, matched fields)]).

        BM25 over name, doc, file and flow summary, plus the exact-name bonus;
        ordered by score, then the stable node order. ``limit`` keeps only the
        best that many, without ordering the rest.
        """
        terms = query_terms(text)
        index = graph.text_index()
        matches = index.bm25.search(terms)
        lowered = text.strip().casefold()
        hits = []
        for key in set(matches).union(index.name_forms.get(lowered, ())):
            entity = graph.entity(key)
            if kinds is not None and entity.get("type") not in kinds:
                continue
            if repo and entity.get("repo") != repo:
                continue
            if file_prefix and not graph.file_of(entity).startswith(file_prefix):
                continue
            if not include_tests and graph.is_test(key):
                continue
            hits.append((key, entity))
        # A storage or association view named after its declaration leaves the
        # name bonus to that declaration when both are hits: the name means
        # the declaration (see _CodeGraphSnapshot.declaration_of).
        hit_set = {key for key, _entity in hits}
        ranked = []
        for key, entity in hits:
            match = matches.get(key)
            score = match.score if match else 0.0
            fields = match.fields if match else ()
            tier = _match_tier(str(entity.get("name") or ""), lowered)
            if tier in _EXACT_NAME_BONUS and graph.declaration_of.get(key) not in hit_set:
                score += _EXACT_NAME_BONUS[tier]
                if "name" not in fields:
                    fields = ("name", *fields)
            if score > 0:
                ranked.append((key, round(score, 4), fields))
        if limit is not None and 0 < limit < len(ranked):
            # Ties at the cut stay in, so the stable order below picks among them.
            cutoff = heapq.nlargest(limit, (score for _key, score, _fields in ranked))[-1]
            ranked = [item for item in ranked if item[1] >= cutoff]
        ranked.sort(key=lambda item: (-item[1], graph.entity_sort_key(item[0])))
        return terms, ranked if limit is None else ranked[:limit]

    @staticmethod
    def _hit(
        graph: _CodeGraphSnapshot, key: str, score: float, fields: tuple[str, ...]
    ) -> dict[str, Any]:
        entity = graph.entity(key)
        node_type = str(entity["type"])
        hit: dict[str, Any] = {
            "id": key,
            "kind": _KIND_BY_TYPE[node_type],
            "type": node_type,
            "name": entity.get("name"),
            "file": graph.file_of(entity) or None,
            "line": entity.get("line"),
            "repo": entity.get("repo"),
        }
        optional = {
            "end_line": entity.get("end_line"),
            "symbol_kind": entity.get("symbol_kind"),
            "doc": graph.doc_of(entity),
        }
        hit.update({name: value for name, value in optional.items() if value not in (None, "")})
        hit["score"] = score
        hit["matched"] = list(fields)
        if key in graph.flows:
            properties = graph.properties_of(entity)
            hit["mechanism"] = properties.get("mechanism")
            hit["rank"] = properties.get("rank")
            hit["summary"] = graph.flow_summary_text(key)
            hit["enola_id"] = entity.get("enola_id")
        return hit

    def _capabilities(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """What this server can do and what this dataset's code graph holds.

        Never raises, also not for a dataset without a code graph (counts are
        then zero). Clients probe it before using the navigation operations.
        """
        symbols = graph.by_type.get("CodeSymbol", set())
        versions = set()
        for repository in graph.repositories.values():
            receipt = _normalize_value(repository.get("last_receipt"))
            version = receipt.get("enola_version") if isinstance(receipt, dict) else None
            if isinstance(version, str) and version:
                versions.add(version)
        return {
            "operation": "capabilities",
            "version": _CAPABILITIES_VERSION,
            "operations": sorted(_OPERATIONS),
            "features": {"docs": bool(graph.documented), "flows": bool(graph.flows)},
            "counts": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "symbols": len(symbols),
                "symbols_with_docs": len(graph.documented & symbols),
                "flows": len(graph.flows),
                "repositories": len(graph.repositories),
            },
            "enola_versions": sorted(versions),
            "limits": copy.deepcopy(_LIMITS),
        }

    def _search(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """Rank facts and flows against free text, without embeddings.

        The text is ``text``, ``query`` or ``q``, else the query text. BM25 over
        each fact's qualified name (weight 3), docstring (2) and file path (1),
        and a flow's summary (1), plus a bonus for an exact or short name
        match. ``kinds`` (default symbol, route, flow, module, storage,
        service; "all" for every kind), ``repo``, ``file_prefix`` and
        ``include_tests`` (default false) narrow it; ``limit`` (10, max 50)
        and ``offset`` page it. No match is an empty ``hits`` list.
        """
        text = self._text_arg(query, "text", "query", "q")
        if not text:
            raise CodeSearchValidationError(
                "CODE search requires text or query text.", param="text"
            )
        kinds = self._search_kinds()
        include_tests = self._include_tests()
        limit = self._limit("search", "limit")
        offset = _offset(self.config.get("offset"))
        terms, ranked = self._text_matches(
            graph,
            text,
            kinds=kinds,
            repo=self.config.get("repo") or None,
            file_prefix=str(self.config.get("file_prefix") or ""),
            include_tests=include_tests,
        )
        page = ranked[offset : offset + limit]
        return {
            "operation": "search",
            "query": text,
            "terms": terms,
            "hits": [self._hit(graph, *item) for item in page],
            "total": len(ranked),
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < len(ranked),
        }

    def _neighbors(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """Every edge of one node, as (relation, direction, neighbour) entries.

        The anchor comes from ``name``, ``id``/``node_id`` or the query text.
        ``direction`` (both; forward lists "out" edges, reverse "in" edges),
        ``relation_types`` and ``kinds`` (neighbour kinds) filter; entries are
        ordered by relation, direction and the stable node order, and paged by
        ``limit`` (50, max 200) and ``offset``. ``flows`` names the flows the
        anchor is the entry, a stage or a route of.
        """
        # Arguments first: a bad one is a usage error whether or not the
        # anchor resolves.
        direction = _direction(self.config.get("direction"), default="both")
        relation_types = self._relation_types()
        kinds = _node_types(self.config.get("kinds"), "kinds")
        limit = self._limit("neighbors", "limit")
        offset = _offset(self.config.get("offset"))
        name = self._identifier_arg("name")
        node_id = self._identifier_arg("id", "node_id")
        repo = self._identifier_arg("repo")
        anchor = graph.resolve(
            name=name or query,
            node_id=node_id,
            repo=repo,
            role="neighbors anchor",
        )

        passes = []
        if direction in {"forward", "both"}:
            passes.append(("out", "target_id", graph.forward.get(anchor, ())))
        if direction in {"reverse", "both"}:
            passes.append(("in", "source_id", graph.reverse.get(anchor, ())))
        entries = []
        for edge_direction, end, edges in passes:
            for edge in edges:
                neighbor_id = edge[end]
                if relation_types is not None and edge["type"] not in relation_types:
                    continue
                if kinds is not None and graph.nodes[neighbor_id].get("type") not in kinds:
                    continue
                entries.append((edge["type"], edge_direction, neighbor_id))
        entries.sort(key=lambda entry: (entry[0], entry[1], graph.node_sort_key(entry[2])))
        page = entries[offset : offset + limit]
        has_more = offset + len(page) < len(entries)
        result: dict[str, Any] = {
            "operation": "neighbors",
            "anchor": graph.fact(anchor, include_relations=False),
            "neighbors": [
                {
                    "relation": relation,
                    "direction": edge_direction,
                    "node": graph.fact(neighbor_id, include_relations=False),
                }
                for relation, edge_direction, neighbor_id in page
            ],
        }
        flows = [
            {"id": flow_id, "name": graph.flows[flow_id].get("name"), "as": role}
            for flow_id, role in graph.flows_by_node.get(anchor, ())
        ]
        if flows:
            result["flows"] = flows
        result.update(
            total=len(entries), offset=offset, limit=limit, has_more=has_more, truncated=has_more
        )
        return result

    def _subgraph(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """The graph around up to 20 seeds, with every cut reported.

        Seeds come from ``seeds``/``names`` (names), ``seed_ids``/``node_ids``/
        ``ids`` (cognee or enola ids), else the query text. ``depth`` (1, max
        3; ``max_depth`` is an alias), ``direction`` (both), ``relation_types``,
        ``kinds`` (node kinds to include; other nodes are walked through but
        not returned, as in explore), ``max_nodes`` (100, max 500; the seeds
        count towards it), ``max_edges`` (300, max 1000) and ``repo`` shape
        it. Seeds that do not resolve go to ``seeds_missing`` with
        candidates; only when none resolves does the call fail.
        ``truncation_reasons`` lists every cut, in order: max_seeds (over 20
        seeds), max_depth (depth over 3), max_nodes and max_edges (a limit was
        reached).
        """
        limits = _LIMITS["subgraph"]
        requested: list[tuple[str, str]] = []
        for field in ("seeds", "names"):
            requested.extend(
                ("name", value) for value in _as_strings(self.config.get(field), field)
            )
        for field in ("seed_ids", "node_ids", "ids"):
            requested.extend(("id", value) for value in _as_strings(self.config.get(field), field))
        if not requested and query.strip():
            requested.append(("name", query.strip()))
        requested = list(dict.fromkeys(requested))
        if not requested:
            raise CodeSearchValidationError(
                "CODE subgraph requires seeds, seed_ids, or query text.", param="seeds"
            )
        reasons = []
        if len(requested) > limits["seeds"]:
            requested = requested[: limits["seeds"]]
            reasons.append("max_seeds")
        depth_field = "depth" if self.config.get("depth") is not None else "max_depth"
        raw_depth = self.config.get(depth_field)
        depth = _count(
            raw_depth,
            field=depth_field,
            default=limits["depth"]["default"],
            maximum=limits["depth"]["max"],
        )
        if (_int_or_none(raw_depth) or 0) > limits["depth"]["max"]:
            reasons.append("max_depth")
        max_nodes = self._limit("subgraph", "max_nodes")
        max_edges = self._limit("subgraph", "max_edges")
        direction = _direction(self.config.get("direction"), default="both")
        relation_types = self._relation_types()
        kinds = _node_types(self.config.get("kinds"), "kinds") or _node_types(
            self.config.get("node_types")
        )
        repo = self.config.get("repo") or None

        seeds: list[str] = []
        missing: list[tuple[str, str, CodeSeedNotFoundError]] = []
        for seed_kind, value in requested:
            try:
                node_id = graph.resolve(
                    name=value if seed_kind == "name" else None,
                    node_id=value if seed_kind == "id" else None,
                    repo=repo,
                    role="subgraph seed",
                )
            except CodeSeedNotFoundError as error:
                missing.append((value, seed_kind, error))
                continue
            if node_id not in seeds:
                seeds.append(node_id)
        if not seeds:
            codes = {error.error_code for _value, _kind, error in missing}
            raise CodeSeedNotFoundError(
                "; ".join(error.message for _value, _kind, error in missing),
                code="ambiguous" if codes == {"ambiguous"} else "not_found",
                param="seeds",
            )

        # max_nodes counts the seeds too: past it, the later seeds are cut.
        node_limit_reached = len(seeds) > max_nodes
        seeds = seeds[:max_nodes]
        included = {seed: 0 for seed in seeds}
        visited = set(seeds)
        queue = deque((seed, 0) for seed in seeds)
        traversed: dict[tuple[str, str, str], dict[str, Any]] = {}
        edges_considered = 0
        depth_reached = 0
        while queue:
            node_id, node_depth = queue.popleft()
            if node_depth >= depth:
                continue
            for neighbor_id, edge in self._incident_edges(graph, node_id, direction):
                if relation_types is not None and edge["type"] not in relation_types:
                    continue
                edges_considered += 1
                traversed.setdefault((edge["source_id"], edge["target_id"], edge["type"]), edge)
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                depth_reached = max(depth_reached, node_depth + 1)
                if kinds is not None and graph.nodes[neighbor_id].get("type") not in kinds:
                    queue.append((neighbor_id, node_depth + 1))
                    continue
                if len(included) >= max_nodes:
                    node_limit_reached = True
                    continue
                included[neighbor_id] = node_depth + 1
                queue.append((neighbor_id, node_depth + 1))
        if node_limit_reached:
            reasons.append("max_nodes")

        # Under the edge limit, keep the edges nearest the seeds.
        edges = sorted(
            (
                edge
                for edge in traversed.values()
                if edge["source_id"] in included and edge["target_id"] in included
            ),
            key=lambda edge: (
                max(included[edge["source_id"]], included[edge["target_id"]]),
                graph.edge_sort_key(edge),
            ),
        )
        if len(edges) > max_edges:
            edges = edges[:max_edges]
            reasons.append("max_edges")
        edges.sort(key=graph.edge_sort_key)

        nodes = sorted(
            included, key=lambda node_id: (included[node_id], graph.node_sort_key(node_id))
        )
        return {
            "operation": "subgraph",
            "seeds": [graph.fact(seed, include_relations=False) for seed in seeds],
            "seeds_missing": [
                self._missing_seed(graph, value, seed_kind, error, repo)
                for value, seed_kind, error in missing
            ],
            "nodes": [
                graph.fact(node_id, depth=included[node_id], include_relations=False)
                for node_id in nodes
            ],
            "edges": [graph.edge_payload(edge) for edge in edges],
            "limits": {
                "seeds": limits["seeds"],
                "depth": depth,
                "max_nodes": max_nodes,
                "max_edges": max_edges,
            },
            "truncated": bool(reasons),
            "truncation_reason": reasons[0] if reasons else None,
            "truncation_reasons": reasons,
            "stats": {
                "nodes_visited": len(visited),
                "edges_considered": edges_considered,
                "depth_reached": depth_reached,
            },
        }

    def _missing_seed(
        self,
        graph: _CodeGraphSnapshot,
        value: str,
        seed_kind: str,
        error: CodeSeedNotFoundError,
        repo: str | None,
    ) -> dict[str, Any]:
        """A seed that did not resolve, with up to five nodes it may have meant."""
        if error.error_code == "ambiguous":
            candidates = graph.name_matches(value, repo=repo)
        elif seed_kind == "name":
            # No node name contains the seed; offer what the text search finds.
            _terms, ranked = self._text_matches(
                graph,
                value,
                kinds=set(_FACT_TYPE_BY_KIND.values()),
                repo=repo,
                file_prefix="",
                include_tests=True,
                limit=5,
            )
            candidates = [key for key, _score, _fields in ranked]
        else:
            candidates = []
        return {
            "seed": value,
            "reason": error.error_code,
            "candidates": [
                {
                    "id": node_id,
                    "name": graph.nodes[node_id].get("name"),
                    "kind": _KIND_BY_TYPE[str(graph.nodes[node_id]["type"])],
                    "file": graph.file_of(graph.nodes[node_id]) or None,
                }
                for node_id in candidates[:5]
            ],
        }

    def _flows(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """Enola's flows: the ways into the program and what each one reaches.

        ``mode`` "list" (the default) ranks them as Enola does (``repo``,
        ``mechanism``, ``include_side`` (false), ``limit`` (20, max 50),
        ``offset``); "show" (the default when ``id``, ``name`` or ``flow`` is
        given) returns one flow with its stages; "through" (the default when
        ``symbol`` or ``through`` is given) lists the flows whose calls reach
        a symbol. A graph without flows raises CodeGraphUnavailableError
        (``feature_unavailable``): they come from Enola's flows.json.
        """
        mode = self.config.get("mode")
        if mode is None:
            if self.config.get("symbol") or self.config.get("through"):
                mode = "through"
            elif self.config.get("id") or self.config.get("name") or self.config.get("flow"):
                mode = "show"
            else:
                mode = "list"
        mode = str(mode).strip().lower()
        if mode not in {"list", "show", "through"}:
            raise CodeSearchValidationError(
                "CODE flows mode must be 'list', 'show', or 'through'.", param="mode"
            )
        # Arguments first, like _neighbors: a bad one is a usage error whether or
        # not this graph has flows. The "no flows" miss below is softened into a
        # per-dataset miss, which would otherwise hide the caller's mistake.
        repo = self._identifier_arg("repo")
        if mode == "list":
            mechanisms = set(_as_strings(self.config.get("mechanism"), "mechanism"))
            include_side = bool(
                _optional_bool(self.config.get("include_side"), field="include_side")
            )
            limit = self._limit("flows", "limit")
            offset = _offset(self.config.get("offset"))
        elif mode == "show":
            self._identifier_arg("id")
            self._identifier_arg("name", "flow")
        else:
            self._identifier_arg("symbol", "through", "name")
            self._identifier_arg("id")
        if not graph.flows:
            if not graph.nodes:
                raise CodeGraphUnavailableError(_NO_CODE_GRAPH_MESSAGE)
            raise CodeGraphUnavailableError(
                _NO_FLOWS_MESSAGE, code="feature_unavailable", param="operation"
            )
        if mode == "show":
            flow_id = self._resolve_flow(graph, query)
            return {"operation": "flows", "mode": "show", "flow": graph.flow_detail(flow_id)}
        if mode == "through":
            return self._flows_through(graph, query)

        matched = []
        for flow_id in graph.flow_order:
            flow = graph.flows[flow_id]
            properties = graph.properties_of(flow)
            if repo and flow.get("repo") != repo:
                continue
            if mechanisms and properties.get("mechanism") not in mechanisms:
                continue
            if not include_side and properties.get("side") is True:
                continue
            matched.append(flow_id)
        page = matched[offset : offset + limit]
        return {
            "operation": "flows",
            "mode": "list",
            "flows": [graph.flow_summary(flow_id) for flow_id in page],
            "total": len(matched),
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < len(matched),
        }

    def _resolve_flow(self, graph: _CodeGraphSnapshot, query: str) -> str:
        """A flow by cognee or Enola id, else by name: exact, then suffix, then substring."""
        repo = self._identifier_arg("repo")
        raw_id = self._identifier_arg("id")
        if raw_id:
            flow_id = raw_id
            if flow_id not in graph.flows:
                flow_id = graph.flow_by_enola_id.get(flow_id, flow_id)
            if flow_id not in graph.flows or (repo and graph.flows[flow_id].get("repo") != repo):
                raise CodeSeedNotFoundError(
                    f"CODE could not resolve flow id {str(raw_id)!r}.", param="id"
                )
            return flow_id

        term = (self._identifier_arg("name", "flow") or query or "").strip()
        if not term:
            raise CodeSearchValidationError(
                "CODE flows show requires a flow id or name.", param="name"
            )
        lowered = term.casefold()
        tiers: dict[int, list[str]] = defaultdict(list)
        for flow_id in graph.flow_order:
            flow = graph.flows[flow_id]
            if repo and flow.get("repo") != repo:
                continue
            name = str(flow.get("name") or "").casefold()
            if name == lowered:
                tiers[3].append(flow_id)
            elif name.endswith(lowered):
                tiers[2].append(flow_id)
            elif lowered in name:
                tiers[1].append(flow_id)
        for tier in (3, 2, 1):
            matches = tiers.get(tier, [])
            if len(matches) == 1:
                return matches[0]
            if matches:
                choices = ", ".join(
                    f"{graph.flows[flow_id].get('name')} "
                    f"[repo={graph.flows[flow_id].get('repo') or '-'}, id={flow_id}]"
                    for flow_id in matches[:10]
                )
                raise CodeSeedNotFoundError(
                    f"CODE flow name {term!r} is ambiguous; use an exact id or repo. "
                    f"Candidates: {choices}",
                    code="ambiguous",
                    param="name",
                )
        raise CodeSeedNotFoundError(f"CODE could not resolve flow name {term!r}.", param="name")

    def _flows_through(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        symbol = graph.resolve(
            name=self._identifier_arg("symbol", "through", "name") or query,
            node_id=self._identifier_arg("id"),
            repo=self._identifier_arg("repo"),
            role="flows symbol",
        )
        reached, truncated = graph.calls_reach(symbol)
        found = graph.flows_reaching(symbol, reached)
        ordered = sorted(
            found, key=lambda flow_id: (found[flow_id][1], graph.flow_rank_key(flow_id))
        )
        flows = [
            {
                **graph.flow_summary(flow_id),
                "via": found[flow_id][0],
                "distance": found[flow_id][1],
            }
            for flow_id in ordered
        ]
        return {
            "operation": "flows",
            "mode": "through",
            "symbol": graph.fact(symbol, include_relations=False),
            "flows": flows,
            "total": len(flows),
            "truncated": truncated,
        }

    def _hubs(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """The most connected nodes: a starting point when no name is known.

        ``kinds`` (default symbol), ``direction`` (both; forward counts
        out-degree, reverse in-degree), ``relation_types`` (default every
        relation except part_of, declares and evidences), ``repo``,
        ``file_prefix``, ``include_tests`` (false: test code is neither listed
        nor counted in a degree, so a function called from many tests is not a
        hub for it) and ``limit`` (20, max 100). Ordered by degree, then the
        stable node order.
        """
        kinds = _node_types(self.config.get("kinds"), "kinds") or {"CodeSymbol"}
        direction = _direction(self.config.get("direction"), default="both")
        relation_types = self._relation_types()
        repo = self.config.get("repo") or None
        file_prefix = str(self.config.get("file_prefix") or "")
        include_tests = self._include_tests()
        limit = self._limit("hubs", "limit")

        def counted(edges: Iterable[Mapping[str, Any]], end: str) -> int:
            return sum(
                1
                for edge in edges
                if (
                    edge["type"] in relation_types
                    if relation_types is not None
                    else edge["type"] not in _CONTAINMENT_EDGE_TYPES
                )
                and (include_tests or edge[end] not in graph.test_nodes)
            )

        ranked = []
        for node_type in kinds:
            for node_id in graph.by_type.get(node_type, ()):
                node = graph.nodes[node_id]
                if repo and node.get("repo") != repo:
                    continue
                if file_prefix and not graph.file_of(node).startswith(file_prefix):
                    continue
                if not include_tests and graph.is_test(node_id):
                    continue
                in_degree = counted(graph.reverse.get(node_id, ()), "source_id")
                out_degree = counted(graph.forward.get(node_id, ()), "target_id")
                degree = {
                    "both": in_degree + out_degree,
                    "forward": out_degree,
                    "reverse": in_degree,
                }[direction]
                if degree:
                    ranked.append((degree, in_degree, out_degree, node_id))
        ranked.sort(key=lambda item: (-item[0], graph.node_sort_key(item[3])))
        return {
            "operation": "hubs",
            "hubs": [
                {
                    **graph.fact(node_id, include_relations=False),
                    "degree": degree,
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                }
                for degree, in_degree, out_degree, node_id in ranked[:limit]
            ],
            "total": len(ranked),
            "limit": limit,
        }

    def _ask(self, graph: _CodeGraphSnapshot, query: str) -> dict[str, Any]:
        """Evidence for a natural-language question about the code, with citations.

        The question is ``question`` or ``text``, else the query text. It is
        searched like ``search`` (``kinds``, ``repo``, ``include_tests``) for
        ``max_seeds`` x 4 hits; the first ``max_seeds`` (5, max 10) fact hits
        are the seeds and flow hits are matched flows. Up to three flows
        follow (matched ones and those whose calls reach the seeds, ranked by
        seeds reached, best seed score, then rank), then the seeds' one-hop
        neighbourhood (callers, callees, route links, the rest; test code only
        with ``include_tests``; ``max_nodes`` (60, max 200) caps it, seeds
        included), then file:line citations (``max_citations``, 25, max 60)
        and a coverage block naming what the graph could not see. Its
        ``seeds`` counts the seeds that can carry a docstring (Enola writes one
        on symbols; a seed of another kind counts when it has one) and
        ``seeds_with_docs`` those that do. Nothing here is written by a model.
        """
        question = self._text_arg(query, "question", "text")
        if not question:
            raise CodeSearchValidationError(
                "CODE ask requires a question or query text.", param="question"
            )
        max_seeds = self._limit("ask", "max_seeds")
        max_nodes = self._limit("ask", "max_nodes")
        max_citations = self._limit("ask", "max_citations")
        repo = self.config.get("repo") or None
        kinds = self._search_kinds()
        include_tests = self._include_tests()
        if not graph.nodes:
            raise CodeGraphUnavailableError(_NO_CODE_GRAPH_MESSAGE)

        terms, ranked = self._text_matches(
            graph,
            question,
            kinds=kinds,
            repo=repo,
            file_prefix=str(self.config.get("file_prefix") or ""),
            include_tests=include_tests,
            limit=max_seeds * 4,
        )
        seed_hits = [item for item in ranked if item[0] in graph.nodes][:max_seeds]
        flow_hits = [item for item in ranked if item[0] in graph.flows]
        seeds = [key for key, _score, _fields in seed_hits]
        documentable = [
            seed
            for seed in seeds
            if seed in graph.documented or graph.nodes[seed].get("type") == "CodeSymbol"
        ]

        flows = self._ask_flows(graph, seed_hits, flow_hits)
        nodes, edges, links, neighborhood_cut = self._ask_neighborhood(
            graph, seeds, max_nodes, include_tests=include_tests
        )
        citations, citations_cut = self._ask_citations(graph, seeds, flows, links, max_citations)

        unresolved = [
            value
            for value in (_int_or_none(flow["metrics"].get("unresolved_calls")) for flow in flows)
            if value is not None
        ]
        unresolved_calls = sum(unresolved) if unresolved else None
        flow_cut = any(flow["metrics"].get("truncated") is True for flow in flows)
        truncated = neighborhood_cut or citations_cut or flow_cut

        notes = []
        if not ranked:
            notes.append(
                f"No symbol, route or flow matched these terms: {', '.join(terms) or '(none)'}. "
                "Try an exact identifier with neighbors, or grep."
            )
        if not graph.documented:
            notes.append("This snapshot has no docstrings, so matching used names and paths only.")
        if not graph.flows:
            notes.append("This snapshot has no flows, so entry points were not traced.")
        if unresolved_calls:
            notes.append(
                f"{unresolved_calls} calls in these flows go to code outside the graph "
                "(libraries or callees the extractor could not resolve)."
            )
        if neighborhood_cut:
            notes.append(
                f"The neighbourhood was cut at {max_nodes} nodes; use subgraph or neighbors "
                "to see more."
            )
        if citations_cut:
            notes.append(f"Citations were cut at {max_citations}.")

        return {
            "operation": "ask",
            "question": question,
            "terms": terms,
            "seeds": [self._hit(graph, *item) for item in seed_hits],
            "flows": flows,
            "neighborhood": {"nodes": nodes, "edges": edges},
            "citations": citations,
            "coverage": {
                "code_graph": True,
                "docs_present": bool(graph.documented),
                "seeds_with_docs": sum(1 for seed in documentable if seed in graph.documented),
                "seeds": len(documentable),
                "flows_present": bool(graph.flows),
                "unresolved_calls": unresolved_calls,
                "truncated": truncated,
                "notes": notes,
            },
            "next": self._ask_next(graph, question, seeds, flows, repo),
            "truncated": truncated,
        }

    @staticmethod
    def _ask_flows(
        graph: _CodeGraphSnapshot,
        seed_hits: list[tuple[str, float, tuple[str, ...]]],
        flow_hits: list[tuple[str, float, tuple[str, ...]]],
    ) -> list[dict[str, Any]]:
        """The matched flows plus the flows through the seeds, best three in show shape."""
        if not graph.flows:
            return []
        reaches: dict[str, list[str]] = {flow_id: [] for flow_id, _score, _fields in flow_hits}
        best: dict[str, float] = {flow_id: score for flow_id, score, _fields in flow_hits}
        for seed, score, _fields in seed_hits:
            reached, _truncated = graph.calls_reach(seed)
            for flow_id in graph.flows_reaching(seed, reached):
                reaches.setdefault(flow_id, []).append(seed)
                best[flow_id] = max(best.get(flow_id, 0.0), score)
        chosen = sorted(
            reaches,
            key=lambda flow_id: (
                -len(reaches[flow_id]),
                -best[flow_id],
                graph.flow_rank_key(flow_id),
            ),
        )[:_ASK_MAX_FLOWS]
        flows = []
        for flow_id in chosen:
            flow = graph.flow_detail(flow_id)
            flow["seeds_reached"] = [graph.nodes[seed].get("name") for seed in reaches[flow_id]]
            flows.append(flow)
        return flows

    @staticmethod
    def _seed_links(
        graph: _CodeGraphSnapshot, seed: str, *, include_tests: bool
    ) -> list[tuple[str, str, str]]:
        """A seed's neighbours as (node, role, relation), most telling first.

        Callers, then callees, then route links (handled_by and the client
        route edges, or any edge to a route), then every other relation
        except containment. Test code is left out unless include_tests.
        """
        callers, callees, routes, others = [], [], [], []
        for role_if_call, end, edges in (
            ("caller", "source_id", graph.reverse.get(seed, ())),
            ("callee", "target_id", graph.forward.get(seed, ())),
        ):
            for edge in edges:
                if edge["type"] in _CONTAINMENT_EDGE_TYPES:
                    continue
                neighbor_id = edge[end]
                if not include_tests and neighbor_id in graph.test_nodes:
                    continue
                if edge["type"] == "calls":
                    (callers if role_if_call == "caller" else callees).append(
                        (neighbor_id, role_if_call, "calls")
                    )
                elif (
                    edge["type"] in _ROUTE_EDGE_TYPES
                    or graph.nodes[neighbor_id].get("type") == "ApiEndpoint"
                ):
                    routes.append((neighbor_id, "neighbor", edge["type"]))
                else:
                    others.append((neighbor_id, "neighbor", edge["type"]))
        return callers + callees + routes + others

    def _ask_neighborhood(
        self,
        graph: _CodeGraphSnapshot,
        seeds: list[str],
        max_nodes: int,
        *,
        include_tests: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[str, str, str]], bool]:
        """One hop around the seeds: (nodes, edges, links, cut).

        Seeds take turns adding their next neighbour, so ``max_nodes`` is
        shared fairly; it counts the seeds too, and past it the later seeds
        are cut. ``links`` maps every added neighbour to the (seed, role,
        relation) that brought it in, for its citation.
        """
        cut = len(seeds) > max_nodes
        seeds = seeds[:max_nodes]
        included = {seed: "seed" for seed in seeds}
        links: dict[str, tuple[str, str, str]] = {}
        queues = [self._seed_links(graph, seed, include_tests=include_tests) for seed in seeds]
        cursors = [0] * len(seeds)
        progressed = True
        while progressed and not cut:
            progressed = False
            for position, seed in enumerate(seeds):
                queue = queues[position]
                while cursors[position] < len(queue) and queue[cursors[position]][0] in included:
                    cursors[position] += 1
                if cursors[position] == len(queue):
                    continue
                if len(included) >= max_nodes:
                    cut = True
                    break
                neighbor_id, role, relation = queue[cursors[position]]
                cursors[position] += 1
                included[neighbor_id] = role
                links[neighbor_id] = (seed, role, relation)
                progressed = True

        nodes = []
        for node_id, role in included.items():
            fact = graph.fact(node_id, depth=0 if role == "seed" else 1, include_relations=False)
            fact["role"] = role
            nodes.append(fact)
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        for seed in seeds:
            for neighbor_id, edge in self._incident_edges(graph, seed, "both"):
                if edge["type"] not in _CONTAINMENT_EDGE_TYPES and neighbor_id in included:
                    edges.setdefault((edge["source_id"], edge["target_id"], edge["type"]), edge)
        ordered_edges = sorted(edges.values(), key=graph.edge_sort_key)
        return nodes, [graph.edge_payload(edge) for edge in ordered_edges], links, cut

    @staticmethod
    def _ask_citations(
        graph: _CodeGraphSnapshot,
        seeds: list[str],
        flows: list[dict[str, Any]],
        links: Mapping[str, tuple[str, str, str]],
        max_citations: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        """What to read before answering: seeds, flow entries and stages, then neighbours.

        One citation per node (the first reason wins), only for nodes with a
        file, at most ``max_citations``; returns (citations, cut).
        """
        citations: list[dict[str, Any]] = []
        seen: set[Any] = set()
        cut = False

        def add(key: Any, why: str, **fields: Any) -> None:
            nonlocal cut
            if not fields["file"] or key in seen:
                return
            if len(citations) >= max_citations:
                cut = True
                return
            seen.add(key)
            citation = {name: fields[name] for name in ("node_id", "kind", "name", "file")}
            for name in ("line", "end_line"):
                if fields[name] is not None:
                    citation[name] = fields[name]
            line = fields["line"]
            citation["cite"] = f"{fields['file']}:{line}" if line is not None else fields["file"]
            if fields["doc"]:
                citation["doc"] = fields["doc"]
            citation["why"] = why
            citations.append(citation)

        def add_node(node_id: str, why: str) -> None:
            node = graph.nodes[node_id]
            add(
                node_id,
                why,
                node_id=node_id,
                kind=_KIND_BY_TYPE[str(node["type"])],
                name=node.get("name"),
                file=graph.file_of(node),
                line=node.get("line"),
                end_line=node.get("end_line"),
                doc=graph.doc_of(node),
            )

        def add_ref(ref: Mapping[str, Any], why: str) -> None:
            if ref.get("node_id") in graph.nodes:
                add_node(ref["node_id"], why)
                return
            # Enola names a symbol the graph does not hold; its file and line
            # still say where to read.
            add(
                ("ref", ref.get("file"), ref.get("line"), ref.get("name")),
                why,
                node_id=None,
                kind="symbol",
                name=ref.get("name"),
                file=ref.get("file"),
                line=ref.get("line"),
                end_line=None,
                doc=ref.get("doc"),
            )

        for seed in seeds:
            add_node(seed, "seed")
        for flow in flows:
            add_ref(flow["entry"], f"entry of flow {flow['name']}")
        for flow in flows:
            for position, stage in enumerate(flow["stages"], 1):
                add_ref(stage, f"stage {stage.get('order') or position} of flow {flow['name']}")
        seed_names = {seed: graph.nodes[seed].get("name") for seed in seeds}
        for node_id, (seed, role, _relation) in links.items():
            if role == "caller":
                add_node(node_id, f"calls {seed_names[seed]}")
            elif role == "callee":
                add_node(node_id, f"called by {seed_names[seed]}")
        for node_id, (seed, role, relation) in links.items():
            if role == "neighbor":
                add_node(node_id, f"linked to {seed_names[seed]} by {relation}")
        return citations, cut

    @staticmethod
    def _ask_next(
        graph: _CodeGraphSnapshot,
        question: str,
        seeds: list[str],
        flows: list[dict[str, Any]],
        repo: str | None,
    ) -> list[dict[str, Any]]:
        """Two to four follow-up code_query calls that widen what ask returned."""
        steps: list[dict[str, Any]] = []
        if flows:
            steps.append(
                {
                    "code_query": {"operation": "flows", "mode": "show", "id": flows[0]["id"]},
                    "why": f"Every stage of flow {flows[0]['name']}.",
                }
            )
        if seeds:
            steps.append(
                {
                    "code_query": {"operation": "neighbors", "id": seeds[0]},
                    "why": f"Every caller, callee and other link of {graph.nodes[seeds[0]].get('name')}.",
                }
            )
            steps.append(
                {
                    "code_query": {"operation": "subgraph", "seed_ids": seeds[:3], "depth": 2},
                    "why": "The graph two hops around the top seeds.",
                }
            )
            return steps
        scope = {"repo": repo} if repo else {}
        if graph.flows and not flows:
            steps.append(
                {
                    "code_query": {"operation": "flows", **scope},
                    "why": "The entry points this snapshot traced, ranked by what they reach.",
                }
            )
        steps.append(
            {
                "code_query": {"operation": "hubs", **scope},
                "why": "The most connected symbols, a place to start when no name matched.",
            }
        )
        if len(steps) < 2:
            steps.append(
                {
                    "code_query": {
                        "operation": "search",
                        "text": question,
                        "kinds": "all",
                        "include_tests": True,
                        **scope,
                    },
                    "why": "The same terms over every kind, test code included.",
                }
            )
        return steps

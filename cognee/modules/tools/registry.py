"""Tool registry: resolves a tool name to a Tool DataPoint + handler callable.

Two storage tiers:
- Built-in tools are registered in-memory at import time. They exist for every
  dataset and do not require cognify to have run.
- User-defined tools live in the graph as Tool DataPoints, written at ingest
  time (e.g. when a Postgres SourceConnection is attached to a dataset) and
  retrieved via the graph engine.
"""

import importlib
from collections.abc import Callable
from typing import Any
from uuid import UUID

from cognee.modules.engine.models import Tool
from cognee.modules.tools.errors import ToolInvocationError, ToolNotFoundError
from cognee.shared.logging_utils import get_logger

ToolHandler = Callable[..., Any]
logger = get_logger("cognee.tools.registry")


_BUILTIN_TOOLS: dict[str, Tool] = {}


# Security invariant: a handler_ref is executable only when its module path lives
# inside one of these packages. Graph-persisted Tool nodes are user-reachable via
# custom graph models, so without this an attacker-supplied node could point a
# handler_ref at any importable module (e.g. "os:system"). Enforced at resolution
# (resolve_handler) *and* discovery (_query_tool_nodes) so an unauthorized tool is
# neither importable nor listable.
ALLOWED_HANDLER_MODULE_PACKAGES: tuple[str, ...] = ("cognee.modules.tools",)


def _module_is_allowed(module_path: str) -> bool:
    return any(
        module_path == package or module_path.startswith(f"{package}.")
        for package in ALLOWED_HANDLER_MODULE_PACKAGES
    )


def validate_handler_ref(handler_ref: str) -> tuple[str, str]:
    """Parse ``handler_ref`` and enforce the allowed handler namespace.

    Returns the ``(module_path, attr)`` pair for the caller to import. Parsing is
    shared with tool discovery so a reference that cannot resolve is never listed.
    Raises ``ToolInvocationError`` when the reference is malformed or unauthorized.
    """
    if ":" in handler_ref:
        module_path, attr = handler_ref.split(":", 1)
    elif "." in handler_ref:
        module_path, _, attr = handler_ref.rpartition(".")
    else:
        raise ToolInvocationError(
            f"handler_ref must be a dotted path or module:attr form, got {handler_ref!r}"
        )
    if not module_path or not attr:
        raise ToolInvocationError(
            f"handler_ref must be a dotted path or module:attr form, got {handler_ref!r}"
        )
    if not _module_is_allowed(module_path):
        raise ToolInvocationError(
            f"handler_ref {handler_ref!r} targets module {module_path!r}, which is outside "
            f"the allowed handler packages {ALLOWED_HANDLER_MODULE_PACKAGES}"
        )
    return module_path, attr


def handler_ref_is_allowed(handler_ref: str) -> bool:
    """True when ``handler_ref`` passes the :func:`validate_handler_ref` invariant."""
    if not isinstance(handler_ref, str):
        return False
    try:
        validate_handler_ref(handler_ref)
    except ToolInvocationError:
        return False
    return True


def register_builtin_tool(tool: Tool) -> None:
    """Register a built-in Tool. Built-ins are global (dataset_id is None)."""
    _BUILTIN_TOOLS[tool.name] = tool


def resolve_handler(handler_ref: str) -> ToolHandler:
    """Import and return the async handler referenced by a dotted path."""
    module_path, attr = validate_handler_ref(handler_ref)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ToolInvocationError(f"Could not import {module_path}: {exc}") from exc

    if not hasattr(module, attr):
        raise ToolInvocationError(f"{module_path} has no attribute {attr!r}")

    handler = getattr(module, attr)
    if not callable(handler):
        raise ToolInvocationError(f"{handler_ref} is not callable")
    return handler


async def get_tool(name: str, dataset_id: UUID | None = None) -> Tool:
    """Look up a Tool by name. Checks built-ins first, then the graph."""
    if name in _BUILTIN_TOOLS:
        return _BUILTIN_TOOLS[name]
    graph_tool = await _find_tool_in_graph(name=name, dataset_id=dataset_id)
    if graph_tool is not None:
        return graph_tool
    raise ToolNotFoundError(
        f"Tool {name!r} is not registered" + (f" for dataset {dataset_id}" if dataset_id else "")
    )


async def list_tools_for_dataset(dataset_id: UUID | None = None) -> list[Tool]:
    """Return every tool visible for a dataset: all built-ins plus graph-scoped tools."""
    tools: list[Tool] = list(_BUILTIN_TOOLS.values())
    tools.extend(await _list_tools_in_graph(dataset_id=dataset_id))
    return tools


async def _find_tool_in_graph(name: str, dataset_id: UUID | None) -> Tool | None:
    """Query the graph for a Tool by name within an optional dataset scope."""
    nodes = await _query_tool_nodes(dataset_id=dataset_id)
    for node in nodes:
        if getattr(node, "name", None) == name:
            return node
    return None


async def _list_tools_in_graph(dataset_id: UUID | None) -> list[Tool]:
    """Return every Tool DataPoint scoped to a dataset (or globally scoped)."""
    return await _query_tool_nodes(dataset_id=dataset_id)


async def _query_tool_nodes(dataset_id: UUID | None) -> list[Tool]:
    """Fetch Tool DataPoints from the graph. Returns [] when the graph is empty
    or no Tool nodes are persisted.

    This path returns [] when graph-backed tools are unavailable, but logs
    backend failures so permission and registry incidents are diagnosable.
    """
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
    except Exception as exc:
        logger.warning(
            "Unable to import graph engine while resolving tools: %s", exc, exc_info=True
        )
        return []

    try:
        graph_engine = await get_graph_engine()
    except Exception as exc:
        logger.warning(
            "Unable to initialize graph engine while resolving tools: %s", exc, exc_info=True
        )
        return []

    get_by_type = getattr(graph_engine, "get_nodes_by_type", None)
    get_graph_data = getattr(graph_engine, "get_graph_data", None)
    if get_by_type is None and get_graph_data is None:
        logger.warning("Graph engine %s cannot list Tool nodes", type(graph_engine).__name__)
        return []

    try:
        if get_by_type is not None:
            raw_nodes = await get_by_type(node_type=Tool)
        else:
            raw_nodes, _ = await get_graph_data()
    except Exception as exc:
        logger.warning("Graph-backed Tool lookup failed: %s", exc, exc_info=True)
        return []
    if isinstance(raw_nodes, tuple) and len(raw_nodes) == 2:
        raw_nodes = raw_nodes[0]

    tools: list[Tool] = []
    for raw in raw_nodes or []:
        tool = _coerce_tool(raw)
        if tool is None:
            continue
        if not handler_ref_is_allowed(tool.handler_ref):
            logger.warning(
                "Skipping graph tool %r: handler_ref %r is outside the allowed packages %s",
                tool.name,
                tool.handler_ref,
                ALLOWED_HANDLER_MODULE_PACKAGES,
            )
            continue
        if dataset_id is None:
            if tool.dataset_id is not None:
                continue
        elif tool.dataset_id not in (None, dataset_id):
            continue
        tools.append(tool)
    return tools


def _coerce_tool(raw) -> Tool | None:
    """Best-effort conversion of a graph-node or vector-payload dict into Tool.

    Graph-stored DataPoint metadata lacks the "type" key required by the MetaData
    TypedDict; strip it before validation and let Pydantic re-derive defaults.
    """
    if isinstance(raw, Tool):
        return raw
    if isinstance(raw, (list, tuple)) and len(raw) > 1:
        raw = raw[1]
    data = raw.model_dump() if hasattr(raw, "model_dump") else raw
    if not isinstance(data, dict):
        return None
    data = {k: v for k, v in data.items() if k != "metadata"}
    try:
        return Tool.model_validate(data)
    except Exception:
        logger.debug("Falling back to None after error in _coerce_tool", exc_info=True)
        return None

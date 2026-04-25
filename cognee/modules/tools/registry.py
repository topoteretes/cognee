"""Tool registry: resolves a tool name to a Tool DataPoint + handler callable.

Two storage tiers:
- Built-in tools are registered in-memory at import time. They exist for every
  dataset and do not require cognify to have run.
- User-defined tools live in the graph as Tool DataPoints, written at ingest
  time (e.g. when a Postgres SourceConnection is attached to a dataset) and
  retrieved via the graph engine.
"""

import importlib
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from cognee.modules.engine.models import Tool
from cognee.modules.tools.errors import ToolInvocationError, ToolNotFoundError
from cognee.shared.logging_utils import get_logger


ToolHandler = Callable[..., Any]
logger = get_logger("cognee.tools.registry")


_BUILTIN_TOOLS: Dict[str, Tool] = {}


def register_builtin_tool(tool: Tool) -> None:
    """Register a built-in Tool. Built-ins are global (dataset_id is None)."""
    _BUILTIN_TOOLS[tool.name] = tool


def resolve_handler(handler_ref: str) -> ToolHandler:
    """Import and return the async handler referenced by a dotted path."""
    if ":" in handler_ref:
        module_path, attr = handler_ref.split(":", 1)
    elif "." in handler_ref:
        module_path, _, attr = handler_ref.rpartition(".")
    else:
        raise ToolInvocationError(
            f"handler_ref must be a dotted path or module:attr form, got {handler_ref!r}"
        )

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


async def get_tool(name: str, dataset_id: Optional[UUID] = None) -> Tool:
    """Look up a Tool by name. Checks built-ins first, then the graph."""
    if name in _BUILTIN_TOOLS:
        return _BUILTIN_TOOLS[name]
    graph_tool = await _find_tool_in_graph(name=name, dataset_id=dataset_id)
    if graph_tool is not None:
        return graph_tool
    raise ToolNotFoundError(
        f"Tool {name!r} is not registered" + (f" for dataset {dataset_id}" if dataset_id else "")
    )


async def list_tools_for_dataset(dataset_id: Optional[UUID] = None) -> List[Tool]:
    """Return every tool visible for a dataset: all built-ins plus graph-scoped tools."""
    tools: List[Tool] = list(_BUILTIN_TOOLS.values())
    tools.extend(await _list_tools_in_graph(dataset_id=dataset_id))
    return tools


async def _find_tool_in_graph(name: str, dataset_id: Optional[UUID]) -> Optional[Tool]:
    """Query the graph for a Tool by name within an optional dataset scope."""
    nodes = await _query_tool_nodes(dataset_id=dataset_id)
    for node in nodes:
        if getattr(node, "name", None) == name:
            return node
    return None


async def _list_tools_in_graph(dataset_id: Optional[UUID]) -> List[Tool]:
    """Return every Tool DataPoint scoped to a dataset (or globally scoped)."""
    return await _query_tool_nodes(dataset_id=dataset_id)


async def _query_tool_nodes(dataset_id: Optional[UUID]) -> List[Tool]:
    """Fetch Tool DataPoints from the graph. Returns [] when the graph is empty
    or no Tool nodes are persisted.

    This path returns [] when graph-backed tools are unavailable, but logs
    backend failures so permission and registry incidents are diagnosable.
    """
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
    except Exception as exc:
        logger.warning("Unable to import graph engine while resolving tools: %s", exc)
        return []

    try:
        graph_engine = await get_graph_engine()
    except Exception as exc:
        logger.warning("Unable to initialize graph engine while resolving tools: %s", exc)
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
        logger.warning("Graph-backed Tool lookup failed: %s", exc)
        return []
    if isinstance(raw_nodes, tuple) and len(raw_nodes) == 2:
        raw_nodes = raw_nodes[0]

    tools: List[Tool] = []
    for raw in raw_nodes or []:
        tool = _coerce_tool(raw)
        if tool is None:
            continue
        if dataset_id is None:
            if tool.dataset_id is not None:
                continue
        elif tool.dataset_id not in (None, dataset_id):
            continue
        tools.append(tool)
    return tools


def _coerce_tool(raw) -> Optional[Tool]:
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
        return None

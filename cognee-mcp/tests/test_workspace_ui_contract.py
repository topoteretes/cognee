"""Pins the workspace UI's tool calls against the server's real tool surface.

The UI is TypeScript calling MCP tools by string name, so nothing at build time
or import time connects the two. When ``b52fcc335`` narrowed the server to the
memory tools, the UI kept calling ``cognify``/``search``/``delete``/
``delete_dataset`` and every one of those buttons raised "Unknown tool" — for
five months, because no test crossed the language boundary.

These tests parse the UI source and check it against ``server.registry``, which
is the only place the two sides can be compared automatically.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

UI_SOURCE = MCP_ROOT / "apps-src" / "src" / "main.tsx"

# `name: "some_tool"` inside a callServerTool({...}) argument object.
TOOL_NAME_RE = re.compile(r'name:\s*"([a-z_]+)"')


def _registered_tool_names() -> set[str]:
    server = importlib.import_module("src.server")
    return set(server.registry.tags)


def _ui_tool_calls() -> set[str]:
    source = UI_SOURCE.read_text(encoding="utf-8")
    # Only the names passed to callServerTool matter; the regex is broad enough
    # to also catch `name:` keys in unrelated object literals, so intersect with
    # the set of strings that appear near a callServerTool call.
    calls = set()
    for match in re.finditer(r"callServerTool\(\s*\{(.{0,400}?)\}\s*\)", source, re.DOTALL):
        calls.update(TOOL_NAME_RE.findall(match.group(1)))
    return calls


def test_ui_source_is_present():
    """The contract tests below are vacuous if the source moved."""
    assert UI_SOURCE.is_file(), f"workspace UI source not found at {UI_SOURCE}"


def test_ui_calls_only_registered_tools():
    """Every tool the workspace UI calls must exist in the server registry.

    A failure here means a workspace button is dead in the host: the call
    returns "Unknown tool" and the UI surfaces it as a generic error.
    """
    ui_calls = _ui_tool_calls()
    assert ui_calls, "parsed no callServerTool names — the regex or the UI shape changed"

    unregistered = sorted(ui_calls - _registered_tool_names())
    assert not unregistered, (
        f"workspace UI calls unregistered tools: {unregistered}. "
        "Either register them with @registry.tool or point the UI at a tool that exists."
    )


@pytest.mark.parametrize(
    "tool_name, required_params",
    [
        # The UI's argument names, which must match the tool signatures. These
        # broke silently once already (the UI sent `search_query` to a `query`
        # parameter), so they are pinned rather than assumed.
        # `background` is load-bearing: ingestion outruns the host's request
        # deadline, so the UI must be able to queue rather than block.
        ("remember", {"data", "dataset_name", "filename", "content_base64", "background"}),
        ("recall", {"query", "search_type", "datasets"}),
        ("forget", {"dataset", "data_id", "dataset_id"}),
        ("list_dataset_data_json", {"dataset_id"}),
        ("create_dataset_json", {"name"}),
        ("visualize_graph_ui", {"dataset_name"}),
    ],
)
def test_tool_accepts_ui_argument_names(tool_name, required_params):
    """Tools must accept the argument names the UI actually sends."""
    import inspect

    server = importlib.import_module("src.server")
    func = getattr(server, tool_name)
    accepted = set(inspect.signature(func).parameters)

    missing = sorted(required_params - accepted)
    assert not missing, (
        f"{tool_name}() does not accept {missing}, which the workspace UI sends. "
        f"Accepted parameters: {sorted(accepted)}."
    )

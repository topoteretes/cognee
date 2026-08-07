"""Flags MCP tools that no test ever exercises.

A tool is registered by decorator, so adding one is easy and forgetting to test
it is easier. At the end of the run this reports every name in
``server.registry`` that was never invoked.

What it measures is invocation, not assertion quality: a tool called incidentally
by an unrelated test counts as covered. That is enough to catch the case this
exists for — a new tool with no test at all — and not enough to catch a test that
calls a tool and asserts nothing.

Warn-only by default. Set ``COGNEE_MCP_STRICT_TOOL_COVERAGE=1`` to fail the run
instead, which is the useful setting for CI. Strict mode only makes sense on a
full ``pytest tests/`` run: a filtered run (``-k``, or a single file) exercises
fewer tools and will report the rest as uncovered.
"""

from __future__ import annotations

import asyncio
import functools
import os
import sys
from pathlib import Path

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

_STRICT_ENV = "COGNEE_MCP_STRICT_TOOL_COVERAGE"

_called: set[str] = set()
_uninstrumented: set[str] = set()


def _instrument(tool, module, name: str) -> None:
    """Route every call to ``name`` through a recorder.

    Both bindings need patching. ``registry.tool`` hands FastMCP the same
    function object it binds at module level, so rebinding one leaves the other
    pointing at the original: ``Tool.fn`` is what a client call reaches (directly
    or via the call_tool proxy) and the module attribute is what a test calling
    ``await server.forget()`` reaches.
    """
    original = tool.fn

    @functools.wraps(original)
    async def recording(*args, **kwargs):
        _called.add(name)
        return await original(*args, **kwargs)

    tool.fn = recording
    setattr(module, name, recording)


def pytest_sessionstart(session):
    import src.server as server

    # No transform is installed at import time, so this is the full catalog.
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}

    for name in server.registry.tags:
        tool = tools.get(name)
        if tool is None or not hasattr(tool, "fn"):
            _uninstrumented.add(name)
            continue
        _instrument(tool, server, name)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    import src.server as server

    registered = set(server.registry.tags)
    uncovered = sorted(registered - _called - _uninstrumented)

    if not uncovered and not _uninstrumented:
        terminalreporter.write_line(
            f"MCP tool coverage: all {len(registered)} registered tools exercised.",
            green=True,
        )
        return

    terminalreporter.section("MCP tool coverage", sep="-", yellow=True)
    for name in uncovered:
        terminalreporter.write_line(f"  {name} has no test coverage", yellow=True)
    for name in sorted(_uninstrumented):
        terminalreporter.write_line(
            f"  {name} could not be instrumented (coverage unknown)", yellow=True
        )
    if uncovered and not os.getenv(_STRICT_ENV):
        terminalreporter.write_line(f"  (set {_STRICT_ENV}=1 to fail on this)")


def pytest_sessionfinish(session, exitstatus):
    """Turn the report into a failure when strict mode is on."""
    if not os.getenv(_STRICT_ENV):
        return

    import src.server as server

    uncovered = set(server.registry.tags) - _called - _uninstrumented
    if uncovered:
        session.exitstatus = 1

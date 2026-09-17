"""The MCP server's own startup recovery call (SDK-577).

A pipeline this same MCP server started and never finished — killed
mid-run — never gets a terminal status on its own, exactly like the API
surface's own startup recovery. Covered here:
- the call happens, scoped to origin "mcp", not every surface's rows
- an older installed cognee with no recovery module degrades to a warning,
  not a crashed boot

cognee-mcp pins a released cognee (pyproject.toml), and the pinned release
this repo's own .venv carries predates cognee.modules.pipelines.recovery
entirely — the exact case _run_startup_recovery's ImportError guard exists
for. So the "recovery is called" case fakes that module in sys.modules
rather than importing the real one, and the "no recovery module" case needs
no faking at all: it is simply today's installed reality.
"""

import importlib
import sys
import types

import pytest

server = importlib.import_module("src.server")


@pytest.mark.asyncio
async def test_startup_recovery_closes_only_this_surfaces_runs(monkeypatch):
    calls = []

    async def _fake_recovery(owned_origins):
        calls.append(owned_origins)

    fake_recovery_module = types.ModuleType("cognee.modules.pipelines.recovery")
    fake_recovery_module.recover_abandoned_pipeline_runs = _fake_recovery
    monkeypatch.setitem(sys.modules, "cognee.modules.pipelines.recovery", fake_recovery_module)

    await server._run_startup_recovery()

    assert calls == [frozenset({"mcp"})]


@pytest.mark.asyncio
async def test_an_older_cognee_with_no_recovery_module_degrades_to_a_warning(monkeypatch, caplog):
    """cognee-mcp pins a released cognee; an installation that predates
    SDK-577 has no cognee.modules.pipelines.recovery at all — including the
    one this test suite itself runs against today. Booting must not fail
    for it — only startup recovery is unavailable."""
    monkeypatch.setitem(sys.modules, "cognee.modules.pipelines.recovery", None)

    with caplog.at_level("WARNING"):
        await server._run_startup_recovery()

    assert any("no startup recovery" in record.message for record in caplog.records)

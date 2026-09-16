"""The MCP server closes the runs it abandoned when it died (SDK-591).

A cognify this server was executing when its container went down leaves a
DATASET_PROCESSING_STARTED row. The API server will not close it: each surface
sweeps only the origin it stamps, because neither can tell the other's dead run
from a live one. So MCP has to close its own, or a dataset whose ingest died
with this container reports "processing" forever.

The call is destructive (it rolls back that run's graph), so where it sits in
main() is part of the contract and is asserted here: after the arguments are
parsed, after migrations, and never when --api-url or --serve-url says this
process does not own the database.

Origin does not mean one process, and stdio is the case that proves it:
stdio is the default transport and it is one process per client, so every
IDE window or agent session against the same local database stamps "mcp"
too. A second session booting while the first is still mid-ingest is not
hypothetical, it is the normal way people use this. Recovery survives that
because the age floor next to the origin check
(COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS, see recovery.py) applies no
matter which owned_origins a caller passes: a STARTED row has to be both
"mcp" and old enough that a session that only just started stopped being
the likely explanation. Nothing about this call site is special-cased for
it, which is the point (github.com/topoteretes/cognee/pull/4983#discussion_r4004955294).
"""

import importlib
import sys
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

server = importlib.import_module("src.server")


class _Stop(Exception):
    """Ends main() once it reaches the transport, so no server is started."""


@pytest.fixture
def run_main(monkeypatch):
    """Drive main() far enough to observe startup, then stop it."""

    async def _run(argv, recovery):
        monkeypatch.setattr(sys, "argv", ["server.py", *argv])

        import cognee.modules.cognify.recovery as recovery_module

        monkeypatch.setattr(
            recovery_module, "recover_stale_cognify_runs_on_startup", recovery, raising=True
        )

        async def _no_migrations():
            return []

        monkeypatch.setattr("cognee.run_migrations.run_migrations", _no_migrations, raising=False)

        async def _stop(*_args, **_kwargs):
            raise _Stop

        monkeypatch.setattr(server, "_serve_with_cors", _stop, raising=False)
        monkeypatch.setattr(server.mcp, "run_stdio_async", _stop, raising=False)

        with pytest.raises(_Stop):
            await server.main()

    return _run


@pytest.mark.asyncio
async def test_startup_closes_runs_this_server_abandoned(run_main):
    """Local mode sweeps the origin it stamps. It does not own the database
    exclusively — stdio is one process per client, so a sibling session can
    stamp "mcp" too — which is why the sweep it calls into also needs the age
    floor covered in cognee/tests/unit/modules/cognify/test_recovery.py rather
    than relying on origin alone."""
    calls = []

    async def _recovery(**kwargs):
        calls.append(kwargs)

    await run_main(["--transport", "stdio"], _recovery)

    assert len(calls) == 1, "MCP must close its own abandoned runs at startup"
    # Its own and nothing else: sweeping "api" here would delete the graph of a
    # run the API server may still be executing.
    assert calls[0]["owned_origins"] == frozenset({"mcp"})


@pytest.mark.asyncio
async def test_a_client_mode_server_sweeps_nothing(run_main):
    """With --api-url this process is a thin HTTP client and the remote API
    owns the rows. Sweeping here would mean deleting graph data out of a
    database this process was explicitly told it does not own."""
    calls = []

    async def _recovery(**kwargs):
        calls.append(kwargs)

    await run_main(
        ["--transport", "stdio", "--api-url", "http://example.invalid"],
        _recovery,
    )

    assert calls == []


@pytest.mark.asyncio
async def test_a_failing_recovery_does_not_stop_the_server(run_main):
    """Recovery is housekeeping. A server that will not boot because it could
    not tidy up is worse than one reporting a stale status."""

    async def _recovery(**_kwargs):
        raise RuntimeError("relational database unreachable")

    # Reaching the transport at all is the assertion: run_main raises _Stop
    # there, which it would never get to if the failure escaped.
    await run_main(["--transport", "stdio"], _recovery)

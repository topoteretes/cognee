"""Regression test for process-hang-on-exit.

Verifies that after a full add/cognify/search cycle (or just after
creating the default database adapters), the process exits cleanly
within a bounded time — i.e. no dangling ThreadPoolExecutor threads,
unclosed httpx clients, or orphan async tasks keep it alive.

Run::

    pytest cognee/tests/unit/test_shutdown_hang.py -v --timeout=30

The ``--timeout=30`` flag is critical: if the test hangs, pytest-timeout
will kill it and report a failure rather than blocking CI forever.
"""

import asyncio
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.timeout(30)
def test_process_exits_after_shutdown():
    """Spawn a child process that creates Cognee adapters and calls shutdown.

    The child must exit within 15 seconds.  If it hangs (the bug this
    patch fixes), the subprocess.run timeout kills it and we fail.
    """
    script = textwrap.dedent("""\
        import asyncio
        import os

        os.environ.setdefault("TELEMETRY_DISABLED", "1")
        os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

        async def main():
            # Import lazily to respect env vars set above
            from cognee.infrastructure.databases.graph import get_graph_engine
            from cognee.infrastructure.databases.vector import get_vector_engine
            from cognee.shutdown import shutdown

            # Force-create the default adapters (Ladybug + LanceDB)
            try:
                graph = await get_graph_engine()
            except Exception:
                pass  # OK if DB init fails in test env

            try:
                vec = get_vector_engine()
            except Exception:
                pass

            # This is the fix: graceful shutdown
            await shutdown()
            print("SHUTDOWN_OK")

        asyncio.run(main())
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]),
    )
    assert "SHUTDOWN_OK" in result.stdout, (
        f"Child did not complete shutdown.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.returncode == 0, (
        f"Child exited with code {result.returncode}.\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_force_exit_watchdog():
    """Verify the force-exit watchdog terminates a stuck process."""
    script = textwrap.dedent("""\
        import threading
        import time

        # Simulate a non-daemon thread that blocks forever (like a stuck
        # ThreadPoolExecutor worker).
        def blocker():
            time.sleep(3600)

        t = threading.Thread(target=blocker, daemon=False)
        t.start()

        # The watchdog should force-exit before the blocker finishes.
        from cognee.shutdown import _start_force_exit_watchdog
        _start_force_exit_watchdog(timeout=2.0)

        # Wait long enough that the watchdog fires.
        time.sleep(5)
        # If we reach here, the watchdog didn't work.
        print("WATCHDOG_FAILED")
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]),
    )
    # The watchdog calls os._exit(0), so the process exits with 0
    # but does NOT print WATCHDOG_FAILED.
    assert "WATCHDOG_FAILED" not in result.stdout, "Watchdog did not fire"

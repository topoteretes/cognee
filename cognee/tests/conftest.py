"""Top-level test config.

Telemetry is off for the whole test tree. ``send_telemetry`` reports unless
``TELEMETRY_DISABLED`` is set or ``ENV`` is ``test``/``dev``, and a plain
``pytest`` run sets neither, so every test session — a contributor's laptop,
a fork's CI — used to count as a real install, and the suite's deliberately
failing fixture tasks as production task errors (gh #5702). ``setdefault``
so an explicit value in the environment still wins, and a test that needs
telemetry on clears the variable itself (see ``test_telemetry_identity``).
Pinned before ``cognee`` is imported so nothing read at import sees the old
value.

``test_subprocess_rss.py`` is a standalone benchmark script, not a pytest
module — its filename starts with ``test_`` for historical reasons but it
parses argparse at import time and imports optional deps (psutil). Skip it
from collection so pytest doesn't crash trying to run it.
"""

import os

os.environ.setdefault("TELEMETRY_DISABLED", "1")

from cognee.shared.logging_utils import get_logger

logger = get_logger()
collect_ignore = ["test_subprocess_rss.py"]


def pytest_sessionfinish(session, exitstatus):
    """Terminate any surviving spawn workers.

    graph_database_subprocess_enabled and vector_db_subprocess_enabled both
    default True, and the workers are daemon=True spawn processes that inherit
    fd 1/2. A worker that outlives the session can hold the runner's output
    pipe open. This is insurance, not the cause: pytest-timeout's thread method
    calls os._exit(1), which bypasses this hook entirely.
    """
    import multiprocessing

    reaped = []
    for child in multiprocessing.active_children():
        try:
            child.terminate()
            child.join(timeout=5)
            reaped.append(child.pid)
        except Exception:
            logger.debug("Ignoring exception in pytest_sessionfinish", exc_info=True)
    if reaped:
        # Say so: a worker alive here outlived every test that could have
        # closed it. The per-test guard in unit/infrastructure/databases/
        # names the test; this is the last line of defence for the rest.
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"Reaped {len(reaped)} leftover subprocess worker(s) at session end: {reaped}",
                yellow=True,
            )

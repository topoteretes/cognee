"""Top-level test config.

``test_subprocess_rss.py`` is a standalone benchmark script, not a pytest
module — its filename starts with ``test_`` for historical reasons but it
parses argparse at import time and imports optional deps (psutil). Skip it
from collection so pytest doesn't crash trying to run it.
"""

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

    for child in multiprocessing.active_children():
        try:
            child.terminate()
            child.join(timeout=5)
        except Exception:
            pass

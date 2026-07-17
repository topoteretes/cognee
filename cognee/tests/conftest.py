"""Top-level test config.

``test_subprocess_rss.py`` is a standalone benchmark script, not a pytest
module — its filename starts with ``test_`` for historical reasons but it
parses argparse at import time and imports optional deps (psutil). Skip it
from collection so pytest doesn't crash trying to run it.
"""

collect_ignore = ["test_subprocess_rss.py"]


import pytest

@pytest.fixture(autouse=True)
async def cleanup_telemetry_after_test():
    yield
    from cognee.shared.utils import close_telemetry
    await close_telemetry()


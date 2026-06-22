"""Top-level test config.

``test_subprocess_rss.py`` is a standalone benchmark script, not a pytest
module — its filename starts with ``test_`` for historical reasons but it
parses argparse at import time and imports optional deps (psutil). Skip it
from collection so pytest doesn't crash trying to run it.
"""

collect_ignore = ["test_subprocess_rss.py"]

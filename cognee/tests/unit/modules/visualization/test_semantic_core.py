"""Run the semantic-map core's Node unit tests inside the pytest suite.

The view's decision layer (``views/semantic_core.js``) is pure and d3-free, so it
is unit-tested in Node via the built-in test runner (no npm, no build step). This
wrapper shells out to ``node --test`` and surfaces failures in the pytest report.
It skips when ``node`` is unavailable (e.g. a Python-only CI), so it never turns
the suite red on environments without Node — it only adds coverage where Node is
present. Node 18+ is required for the built-in test runner.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_TEST_JS = Path(__file__).parent / "semantic_core.test.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_semantic_core_node_suite():
    result = subprocess.run(
        ["node", "--test", str(_TEST_JS)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail("semantic_core.js Node tests failed:\n" + result.stdout + "\n" + result.stderr)

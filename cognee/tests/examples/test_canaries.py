"""Canary tests for the mocked-example harness.

These three exercise the harness across increasing pipeline depth so a broken
mock is caught immediately, before the per-folder suites are added:

* ``recall_core``                 -- high-level remember -> recall API.
* ``simple_cognee_example``       -- full cognify + GRAPH_COMPLETION recall.
* ``custom_cognify_pipeline``     -- low-level custom Task pipeline + search.

Each loads the real script from ``examples/`` and awaits its ``main()``; the
assertion is simply that it runs to completion with no exception under the
mocked LLM + embedding layers (no API key, no network).
"""

from __future__ import annotations

import pytest

from cognee.tests.utils.example_runner import import_example

pytestmark = pytest.mark.asyncio


async def test_recall_core(isolated_example_env):
    module = import_example("examples/guides/recall_core.py")
    await module.main()


async def test_simple_cognee_example(isolated_example_env):
    module = import_example("examples/demos/simple_cognee_example.py")
    await module.main()


async def test_custom_cognify_pipeline_example(isolated_example_env):
    module = import_example("examples/custom_pipelines/custom_cognify_pipeline_example.py")
    await module.main()

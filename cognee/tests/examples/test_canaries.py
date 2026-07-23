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

from cognee.tests.utils.example_runner import import_example, invoke_example_main

pytestmark = pytest.mark.asyncio


async def test_recall_core(isolated_example_env):
    rel_path = "examples/guides/recall_core.py"
    module = import_example(rel_path)
    await invoke_example_main(module, rel_path, work_dir=isolated_example_env)


async def test_simple_cognee_example(isolated_example_env):
    rel_path = "examples/demos/simple_cognee_example.py"
    module = import_example(rel_path)
    await invoke_example_main(module, rel_path, work_dir=isolated_example_env)


async def test_custom_cognify_pipeline_example(isolated_example_env):
    rel_path = "examples/custom_pipelines/custom_cognify_pipeline_example.py"
    module = import_example(rel_path)
    await invoke_example_main(module, rel_path, work_dir=isolated_example_env)

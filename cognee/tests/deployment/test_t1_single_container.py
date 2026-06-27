"""T1 — single-container API e2e golden flow over HTTP."""

import pytest

from cognee.tests.deployment.helpers import assert_no_tracebacks_in_logs, fetch_container_logs


@pytest.mark.deployment
@pytest.mark.asyncio
async def test_t1_single_container_golden_flow(running_container, api_client, golden_flow):
    await golden_flow(api_client)

    stdout, stderr = fetch_container_logs(running_container["container_name"])
    assert_no_tracebacks_in_logs(stdout, stderr)

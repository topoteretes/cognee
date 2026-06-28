import pytest
from cognee.tests.deployment.flows.golden_flow import golden_flow

@pytest.mark.asyncio
@pytest.mark.deployment
async def test_harness_works(api_client, mock_llm_server):
    """Proves the deployment harness is functional in CI"""
    result = await golden_flow(api_client)
    assert result is True, "golden_flow failed"

import pytest

@pytest.mark.deployment
@pytest.mark.asyncio
async def test_deployment_api_golden_flow(api_client, run_golden_flow):
    dataset_name = "test_deployment_dataset"
    await run_golden_flow(api_client, dataset_name)

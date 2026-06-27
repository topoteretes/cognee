import pytest
import subprocess

@pytest.mark.deployment
@pytest.mark.asyncio
async def test_t1_single_container_golden_flow(running_container, api_client, run_golden_flow):
    dataset_name = "test_deployment_dataset"
    
    # 1. Run the integration golden flow using the conftest.py runner helper
    await run_golden_flow(api_client, dataset_name)
    
    # 2. Assert container logs contain no Python tracebacks
    container_name = running_container["container_name"]
    proc = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True, errors="replace")
    
    assert "Traceback" not in proc.stdout, f"Traceback found in container stdout logs:\n{proc.stdout}"
    assert "Traceback" not in proc.stderr, f"Traceback found in container stderr logs:\n{proc.stderr}"

"""Fly.io deployment end-to-end test."""

import os
import uuid

import pytest

from cognee.tests.deployment.fly_harness import (
    authed_client,
    deploy_fly_app,
    destroy_fly_app,
    golden_flow,
    wait_for_health,
)

REQUIRED_SECRETS = ("FLY_API_TOKEN", "LLM_API_KEY")
HEALTH_TIMEOUT = int(os.getenv("FLY_HEALTH_TIMEOUT", "600"))


def _missing_secrets() -> list[str]:
    return [name for name in REQUIRED_SECRETS if not os.getenv(name)]


@pytest.mark.deployment
@pytest.mark.asyncio
async def test_fly_deploy_golden_flow():
    missing = _missing_secrets()
    if missing:
        pytest.skip(f"Missing required secret(s): {', '.join(missing)}")

    run_id = os.getenv("GITHUB_RUN_ID") or uuid.uuid4().hex[:8]
    app_name = f"cognee-ci-{run_id}"

    llm_env = {"LLM_API_KEY": os.getenv("LLM_API_KEY")}

    try:
        url = deploy_fly_app(
            app_name,
            region=os.getenv("FLY_REGION", "iad"),
            volume_size=int(os.getenv("FLY_VOLUME_SIZE", "1")),
            llm_env=llm_env,
        )

        wait_for_health(f"{url}/health", timeout=HEALTH_TIMEOUT, interval=10)

        async with authed_client(url) as client:
            assert await golden_flow(client) is True
    finally:
        destroy_fly_app(app_name)

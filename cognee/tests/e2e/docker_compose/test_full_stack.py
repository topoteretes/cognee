import asyncio
from uuid import uuid4

import httpx

from .conftest import compose_command, wait_for_http


GOLDEN_FILE_NAME = "golden-flow"
GOLDEN_FILE_EXTENSION = "txt"
GOLDEN_FILE_UPLOAD = f"{GOLDEN_FILE_NAME}.{GOLDEN_FILE_EXTENSION}"


def _dataset_by_name(client: httpx.Client, dataset_name: str) -> dict:
    response = client.get("/api/v1/datasets")
    response.raise_for_status()

    for dataset in response.json():
        if dataset["name"] == dataset_name:
            return dataset

    raise AssertionError(f"Dataset {dataset_name!r} was not returned by the API")


def _has_golden_file(data_items: list[dict]) -> bool:
    return any(
        item["name"] == GOLDEN_FILE_NAME and item["extension"] == GOLDEN_FILE_EXTENSION
        for item in data_items
    )


def golden_flow(client: httpx.Client) -> dict:
    """Small deterministic full-stack flow for compose CI.

    The #3358 deployment harness will eventually provide the shared LLM-backed
    golden_flow(). This local flow intentionally avoids LLM calls so the
    PR-blocking docker-compose job stays keyless while still proving API,
    storage, and Postgres-backed persistence.
    """

    dataset_name = f"docker-compose-e2e-{uuid4().hex}"
    payload = b"Cognee docker-compose E2E verifies persistent AI memory plumbing."

    create_response = client.post("/api/v1/datasets", json={"name": dataset_name})
    create_response.raise_for_status()
    dataset = create_response.json()

    add_response = client.post(
        "/api/v1/add",
        data={"datasetName": dataset_name, "run_in_background": "false"},
        files={"data": (GOLDEN_FILE_UPLOAD, payload, "text/plain")},
        timeout=120.0,
    )
    add_response.raise_for_status()

    data_response = client.get(f"/api/v1/datasets/{dataset['id']}/data")
    data_response.raise_for_status()
    data_items = data_response.json()

    assert data_items, "Golden flow data item was not persisted"
    assert _has_golden_file(data_items)

    return {"dataset_name": dataset_name, "dataset_id": dataset["id"]}


def assert_dataset_persisted(client: httpx.Client, dataset_name: str) -> None:
    dataset = _dataset_by_name(client, dataset_name)

    data_response = client.get(f"/api/v1/datasets/{dataset['id']}/data")
    data_response.raise_for_status()

    assert _has_golden_file(data_response.json())


async def call_mcp_tool(mcp_base_url: str) -> None:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(f"{mcp_base_url}/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool("list_datasets_json", {})

    assert not getattr(result, "isError", False), result


def test_docker_compose_full_stack(api_base_url: str, mcp_base_url: str):
    with httpx.Client(base_url=api_base_url, timeout=30.0) as client:
        health_response = client.get("/health")
        health_response.raise_for_status()

        golden = golden_flow(client)

        mcp_health = httpx.get(f"{mcp_base_url}/health", timeout=10.0)
        mcp_health.raise_for_status()
        asyncio.run(call_mcp_tool(mcp_base_url))

        compose_command("restart", "postgres", "cognee", "cognee-mcp")
        compose_command("up", "-d", "--wait", "--wait-timeout", "300")
        wait_for_http(f"{api_base_url}/health")
        wait_for_http(f"{mcp_base_url}/health")

        assert_dataset_persisted(client, golden["dataset_name"])

"""Reusable add → cognify → search flow for deployment e2e tests."""

from __future__ import annotations

import asyncio
import uuid

import httpx

from cognee.tests.deployment.mock_llm import GOLDEN_DOCUMENT, GOLDEN_ENTITY

PIPELINE_COMPLETED = "DATASET_PROCESSING_COMPLETED"
PIPELINE_ERRORED = "DATASET_PROCESSING_ERRORED"


async def register_and_login(
    client: httpx.AsyncClient,
    email: str | None = None,
    password: str = "test_password",
) -> str:
    if email is None:
        email = f"deploy_test_{uuid.uuid4().hex[:8]}@example.com"

    reg_payload = {
        "email": email,
        "password": password,
        "is_active": True,
        "is_superuser": False,
        "is_verified": False,
    }
    try:
        await client.post("/api/v1/auth/register", json=reg_payload)
    except Exception:
        pass

    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return token


async def run_golden_flow(client: httpx.AsyncClient, dataset_name: str | None = None) -> str:
    """Drive add → cognify → poll → search and assert the golden entity is retrievable."""
    if dataset_name is None:
        dataset_name = f"deploy_golden_{uuid.uuid4().hex[:8]}"

    await register_and_login(client)

    files = {
        "data": ("document.txt", GOLDEN_DOCUMENT.encode("utf-8"), "text/plain"),
    }
    resp = await client.post("/api/v1/add", data={"datasetName": dataset_name}, files=files)
    resp.raise_for_status()

    resp = await client.get("/api/v1/datasets")
    resp.raise_for_status()
    datasets = resp.json()

    dataset_id = next((ds["id"] for ds in datasets if ds["name"] == dataset_name), None)
    assert dataset_id is not None, f"Dataset {dataset_name} not found in: {datasets}"

    resp = await client.post(
        "/api/v1/cognify",
        json={"datasets": [dataset_name], "run_in_background": True},
    )
    resp.raise_for_status()

    status_completed = False
    for _ in range(120):
        resp = await client.get(f"/api/v1/datasets/status?dataset={dataset_id}")
        resp.raise_for_status()
        status_data = resp.json()
        status = status_data.get(str(dataset_id))
        if status == PIPELINE_COMPLETED:
            status_completed = True
            break
        if status == PIPELINE_ERRORED:
            raise RuntimeError(
                f"Cognify pipeline failed for dataset {dataset_id}. Status payload: {status_data}"
            )
        await asyncio.sleep(1)

    assert status_completed, f"Cognify pipeline did not complete in time for dataset {dataset_id}"

    resp = await client.post(
        "/api/v1/search",
        json={
            "search_type": "GRAPH_COMPLETION",
            "query": "Who developed relativity?",
            "dataset_ids": [dataset_id],
        },
    )
    resp.raise_for_status()
    results_gc = resp.json()
    assert any(GOLDEN_ENTITY in str(item) for item in results_gc), (
        f"{GOLDEN_ENTITY} not found in GRAPH_COMPLETION results: {results_gc}"
    )

    resp = await client.post(
        "/api/v1/search",
        json={
            "search_type": "CHUNKS",
            "query": GOLDEN_ENTITY,
            "dataset_ids": [dataset_id],
        },
    )
    resp.raise_for_status()
    results_chunks = resp.json()
    assert any(GOLDEN_ENTITY in str(item) for item in results_chunks), (
        f"{GOLDEN_ENTITY} not found in CHUNKS results: {results_chunks}"
    )

    return dataset_id

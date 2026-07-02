"""
End-to-end deployment test for the Helm chart on a kind cluster (T7 #3365).

Deploys the chart and runs the golden flow (add-cognify-search) over HTTP
against the live cognee Service, using an in-cluster mock LLM so no real API key
is needed and the result is deterministic (the mock always returns "Alice").

Uses a local golden_flow() for now; will switch to the shared harness flow
(#3358 / #3596) once that merges.

Run with: pytest -m deployment_graph
"""
import os
import time

import httpx
import pytest

pytestmark = [pytest.mark.deployment, pytest.mark.deployment_graph]

BASE_URL = os.environ.get("COGNEE_BASE_URL", "http://localhost:8000")
DATASET = os.environ.get("COGNEE_TEST_DATASET", "helm_kind_e2e")
MOCK_ENTITY = os.environ.get("MOCK_ENTITY", "Alice")
DOC_TEXT = f"{MOCK_ENTITY} works at Cognee and manages the AI memory platform."

# cognify + embedding over the mock is quick, but pod scheduling/boot is not.
# keep timeouts generous so CI on a small runner isnt flaky.
REQUEST_TIMEOUT = 120.0


def wait_for_health(url: str, timeout: float = 300.0, interval: float = 3.0) -> None:
    """Poll GET <url> until it returns 200, or raise on timeout.

    Local mirror of the harness health poller; dropped for the shared
    helpers/health.py once T0 (#3596) lands.
    """
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=10.0)
            if r.status_code == 200:
                return
            last_err = f"status {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(interval)
    raise TimeoutError(f"health not ready at {url} after {timeout}s (last: {last_err})")


async def golden_flow(api_client: httpx.AsyncClient) -> bool:
    """Deployed golden flow: add -> cognify -> search, asserting the seeded entity.
    Signature mirrors the frozen harness flow (#3596): takes an async httpx
    client bound to the cognee base URL. Kept intentionally minimal and matched
    to the endpoint shapes verified against the running container:
      - add: multipart/form-data with `data` as an uploaded file
      - cognify: blocking (returns PipelineRunCompleted synchronously), so no
        status polling is needed for the mock path
      - search: GRAPH_COMPLETION, asserting the deterministic mock entity
    """
    # 1. add (multipart file upload)
    files = {"data": ("doc.txt", DOC_TEXT.encode("utf-8"), "text/plain")}
    add_resp = await api_client.post(
        "/api/v1/add", files=files, data={"datasetName": DATASET}
    )
    assert add_resp.status_code == 200, f"add failed: {add_resp.status_code} {add_resp.text}"
    assert "PipelineRunCompleted" in add_resp.text, f"add did not complete: {add_resp.text}"

    # 2. cognify (blocking)
    cognify_resp = await api_client.post("/api/v1/cognify", json={"datasets": [DATASET]})
    assert cognify_resp.status_code == 200, (
        f"cognify failed: {cognify_resp.status_code} {cognify_resp.text}"
    )
    assert "PipelineRunCompleted" in cognify_resp.text, (
        f"cognify did not complete: {cognify_resp.text}"
    )

    # 3. search (GRAPH_COMPLETION)
    search_resp = await api_client.post(
        "/api/v1/search",
        json={
            "search_type": "GRAPH_COMPLETION",
            "query": "Who works at Cognee?",
            "datasets": [DATASET],
        },
    )
    assert search_resp.status_code == 200, (
        f"search failed: {search_resp.status_code} {search_resp.text}"
    )

    results = search_resp.json()
    flat = " ".join(map(str, results)) if isinstance(results, list) else str(results)
    assert MOCK_ENTITY in flat, f"expected '{MOCK_ENTITY}' in search results, got: {results}"
    return True


@pytest.mark.asyncio
async def test_golden_flow_on_helm_kind():
    """Full deployed golden flow against the Helm/kind deployment."""
    wait_for_health(f"{BASE_URL}/health")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=REQUEST_TIMEOUT) as client:
        assert await golden_flow(client) is True
"""The golden flow exercised against the running cognee API on :8000.

``golden_flow()`` walks the core public contract a real client depends on:

    health  ->  login  ->  add  ->  list datasets  ->  list data  ->  status

and, only when explicitly asked (a real LLM key is configured), the heavier
``cognify -> search`` leg. The ingestion leg needs no LLM, so the default
PR-blocking run stays fully mocked.

The function returns a :class:`GoldenFlowResult` so callers (the persistence
test) can re-check that the same dataset is still there after a restart.
"""

from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

import requests

from config import CONFIG


@dataclass
class GoldenFlowResult:
    token: str
    dataset_name: str
    dataset_id: str
    data_count: int
    searched: bool = False


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def login(api_url: str, username: str, password: str) -> str:
    """Authenticate against the seeded default account and return a bearer token."""
    response = requests.post(
        f"{api_url}/api/v1/auth/login",
        data={"username": username, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    assert token, f"login returned no access_token: {response.text}"
    return token


def check_liveness(api_url: str) -> None:
    """Root + health endpoints behave as the public contract promises."""
    root = requests.get(f"{api_url}/", timeout=15)
    assert root.status_code == 200, f"GET / -> {root.status_code}"
    assert root.json().get("message") == "Hello, World, I am alive!", root.text

    health = requests.get(f"{api_url}/health", timeout=15)
    assert health.status_code == 200, f"GET /health -> {health.status_code}"


def add_text(api_url: str, token: str, dataset_name: str, text: str) -> dict:
    """Upload an in-memory text document to a dataset (synchronous ingestion)."""
    files = {"data": (f"{dataset_name}.txt", io.BytesIO(text.encode("utf-8")), "text/plain")}
    response = requests.post(
        f"{api_url}/api/v1/add",
        headers=_auth_header(token),
        data={"datasetName": dataset_name, "run_in_background": "false"},
        files=files,
        timeout=120,
    )
    assert response.status_code in (200, 201), f"add -> {response.status_code}: {response.text}"
    return response.json()


def find_dataset(api_url: str, token: str, dataset_name: str) -> Optional[dict]:
    response = requests.get(f"{api_url}/api/v1/datasets", headers=_auth_header(token), timeout=30)
    response.raise_for_status()
    for dataset in response.json():
        if dataset.get("name") == dataset_name:
            return dataset
    return None


def wait_for_dataset(
    api_url: str, token: str, dataset_name: str, *, timeout: float = 120.0
) -> dict:
    """Poll the datasets listing until our dataset shows up after an add."""
    deadline = time.monotonic() + timeout
    last: Optional[dict] = None
    while time.monotonic() < deadline:
        last = find_dataset(api_url, token, dataset_name)
        if last is not None:
            return last
        time.sleep(2)
    raise AssertionError(f"dataset '{dataset_name}' never appeared after add")


def list_dataset_data(api_url: str, token: str, dataset_id: str) -> List[dict]:
    response = requests.get(
        f"{api_url}/api/v1/datasets/{dataset_id}/data",
        headers=_auth_header(token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def cognify_and_search(api_url: str, token: str, dataset_name: str) -> None:
    """LLM-dependent leg: build the graph and run one search against it."""
    cognify = requests.post(
        f"{api_url}/api/v1/cognify",
        headers=_auth_header(token),
        json={"datasets": [dataset_name], "run_in_background": False},
        timeout=900,
    )
    assert cognify.status_code in (200, 201), f"cognify -> {cognify.status_code}: {cognify.text}"

    search = requests.post(
        f"{api_url}/api/v1/search",
        headers=_auth_header(token),
        json={
            "query": "What is this document about?",
            "search_type": "GRAPH_COMPLETION",
            "datasets": [dataset_name],
        },
        timeout=300,
    )
    assert search.status_code == 200, f"search -> {search.status_code}: {search.text}"
    assert search.json(), "search returned an empty result"


def golden_flow(
    api_url: Optional[str] = None,
    *,
    token: Optional[str] = None,
    dataset_name: Optional[str] = None,
    run_llm: Optional[bool] = None,
) -> GoldenFlowResult:
    """Run the end-to-end golden flow against the API and return what was created."""
    api_url = (api_url or CONFIG.api_url).rstrip("/")
    run_llm = CONFIG.run_llm if run_llm is None else run_llm
    dataset_name = dataset_name or f"e2e_{uuid.uuid4().hex[:8]}"

    check_liveness(api_url)
    token = token or login(api_url, CONFIG.username, CONFIG.password)

    add_text(
        api_url,
        token,
        dataset_name,
        "Cognee turns documents into a queryable knowledge graph. "
        "This text is ingested by the docker-compose end-to-end test.",
    )

    dataset = wait_for_dataset(api_url, token, dataset_name)
    dataset_id = str(dataset["id"])

    data = list_dataset_data(api_url, token, dataset_id)
    assert data, f"dataset '{dataset_name}' has no data items after add"

    searched = False
    if run_llm:
        cognify_and_search(api_url, token, dataset_name)
        searched = True

    return GoldenFlowResult(
        token=token,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        data_count=len(data),
        searched=searched,
    )

"""Full-stack end-to-end test for the docker-compose deployment.

Replaces the old ``up -d -> sleep 30 -> down`` placeholder in
``.github/workflows/docker_compose.yml`` with a real assertion suite:

* the golden flow against the cognee API on :8000,
* a real MCP tool call against the cognee-mcp service on :8001,
* Postgres-backed persistence across a container recreate, and
* a traceback scan over every service's logs.

By default the LLM-dependent leg (cognify/search) is skipped, so the
PR-blocking run never calls a real model ("mock LLM by default"). Set
``COGNEE_E2E_RUN_LLM=1`` with a real key to exercise it.
"""

from __future__ import annotations

import time

from compose_utils import recreate_service, service_logs, wait_for_http_ok
from config import CONFIG
from golden_flow import find_dataset, golden_flow, login
from mcp_client import call_mcp_tool

# Services whose logs must be free of Python tracebacks.
LOG_SERVICES = ("cognee", "cognee-mcp", "postgres")


def test_golden_flow_api(api_ready):
    """health -> login -> add -> datasets -> data (+ optional cognify/search)."""
    result = golden_flow(api_ready)

    assert result.dataset_id
    assert result.data_count >= 1
    if CONFIG.run_llm:
        assert result.searched, "LLM run requested but search leg did not execute"


def test_mcp_health_and_tool_call(mcp_ready):
    """MCP service is healthy and a real tool call returns structured content."""
    health = wait_for_http_ok(CONFIG.mcp_health_url, name="cognee-mcp /health")
    assert health.json().get("status") == "ok", health.text

    call = call_mcp_tool()
    # `list_datasets_json` returns {"datasets": [...]} in structuredContent.
    assert call.structured is not None, "MCP tool returned no structured content"
    assert "datasets" in call.structured, call.structured
    assert isinstance(call.structured["datasets"], list)


def test_postgres_persistence_across_recreate(requires_compose, api_ready):
    """Data added through the API survives a Postgres container recreate.

    A force-recreate discards the container's writable layer, so the dataset
    only survives if Postgres data lives on the named volume. This makes the
    volume requirement explicit: comment the volume out and this test fails.
    """
    result = golden_flow(api_ready)

    recreate_service("postgres")

    # Postgres comes back on a fresh container; wait for it (and the API) again.
    wait_for_http_ok(CONFIG.health_url, name="cognee API after postgres recreate")

    token = login(api_ready, CONFIG.username, CONFIG.password)
    # The app may briefly re-establish its connection pool after the DB bounced.
    dataset = None
    for _ in range(20):
        dataset = find_dataset(api_ready, token, result.dataset_name)
        if dataset is not None:
            break
        time.sleep(3)

    assert dataset is not None, (
        f"dataset '{result.dataset_name}' did not survive the Postgres recreate — "
        "the postgres_data volume is likely missing from docker-compose.yml"
    )


def test_service_logs_are_traceback_free(requires_compose):
    """No service should have emitted an unhandled Python traceback."""
    offenders = {}
    for service in LOG_SERVICES:
        logs = service_logs(service)
        if "Traceback (most recent call last)" in logs:
            tail = "\n".join(logs.splitlines()[-40:])
            offenders[service] = tail

    assert not offenders, "Tracebacks found in service logs:\n" + "\n\n".join(
        f"--- {svc} ---\n{tail}" for svc, tail in offenders.items()
    )

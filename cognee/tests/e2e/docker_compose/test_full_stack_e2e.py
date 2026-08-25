"""Full-stack end-to-end test for the docker-compose deployment.

Complements the workflow-level remember/recall smoke test in
``.github/workflows/docker_compose.yml`` with a real assertion suite:

* the golden flow against the cognee API on :8000,
* a real MCP tool call against the cognee-mcp service on :8001,
* a traceback scan over every service's logs, and
* Postgres-backed persistence across a container recreate.

By default the LLM-dependent leg (cognify/search) is skipped, so this suite
never calls a real model ("mock LLM by default"). Set ``COGNEE_E2E_RUN_LLM=1``
with a real key to exercise it.

Ordering matters: the traceback scan runs *before* the persistence test,
because force-recreating Postgres legitimately drops the API's live database
connections and the resulting (recovered-from) errors may be logged with
tracebacks.
"""

from __future__ import annotations

import time

import requests

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
    """MCP service is healthy and a real tool call round-trips."""
    health = wait_for_http_ok(CONFIG.mcp_health_url, name="cognee-mcp /health")
    assert health.json().get("status") == "ok", health.text

    # `cognify_status` (no dataset_name) reports on the agent-scoped default
    # dataset's cognify_pipeline runs; on a fresh dataset that is an empty or
    # "not started"-style status rather than a fixed string, so `call_mcp_tool`
    # already asserts the call round-tripped without an MCP error — here we
    # only check it produced some text.
    call = call_mcp_tool()
    assert call.result_text.strip(), "cognify_status returned no text"


def test_service_logs_are_traceback_free(requires_compose):
    """No service should have emitted an unhandled Python traceback.

    Must run before the persistence test: recreating Postgres drops the API's
    live connections, and those transient (handled) failures can be logged
    with tracebacks.
    """
    offenders = {}
    for service in LOG_SERVICES:
        logs = service_logs(service)
        if "Traceback (most recent call last)" in logs:
            tail = "\n".join(logs.splitlines()[-40:])
            offenders[service] = tail

    assert not offenders, "Tracebacks found in service logs:\n" + "\n\n".join(
        f"--- {svc} ---\n{tail}" for svc, tail in offenders.items()
    )


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

    # The app re-establishes its connection pool after the DB bounced, so the
    # first requests may fail transiently — keep retrying until the deadline.
    deadline = time.monotonic() + 90
    dataset = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            token = login(api_ready, CONFIG.username, CONFIG.password)
            dataset = find_dataset(api_ready, token, result.dataset_name)
            if dataset is not None:
                break
        except (requests.RequestException, AssertionError) as exc:
            last_error = exc
        time.sleep(3)

    assert dataset is not None, (
        f"dataset '{result.dataset_name}' did not survive the Postgres recreate — "
        "the postgres_data volume is likely missing from docker-compose.yml "
        f"(last error: {last_error!r})"
    )

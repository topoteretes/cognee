"""Backend-DB matrix golden-flow tests (T10, issue #3368).

Runs the same ``add -> cognify -> search`` golden flow and a
``remember -> recall`` flow against a running API container backed by each
supported DB stack, asserting the same entity is retrievable regardless of
backend.

The free, file-/Postgres-based stacks are PR-blocking. The Neo4j stack is
marked ``nightly`` because it needs heavier external infrastructure.
"""

from __future__ import annotations

import pytest

from cognee.tests.deployment.golden_flow import run_golden_flow, run_remember_recall_flow

pytestmark = pytest.mark.deployment

# Free backends — PR-blocking.
FREE_STACKS = ["sqlite_lancedb_kuzu", "postgres_pgvector_postgresgraph"]
# Heavier external infra — nightly only.
NIGHTLY_STACKS = ["neo4j_postgres"]


@pytest.mark.asyncio
@pytest.mark.parametrize("running_container", FREE_STACKS, indirect=True)
async def test_golden_flow(api_client, running_container):
    """add -> cognify -> search retrieves the golden entity on each free backend."""
    await run_golden_flow(api_client)


@pytest.mark.asyncio
@pytest.mark.parametrize("running_container", FREE_STACKS, indirect=True)
async def test_remember_recall(api_client, running_container):
    """remember -> recall (different dataset) retrieves the golden entity on each free backend."""
    await run_remember_recall_flow(api_client)


@pytest.mark.nightly
@pytest.mark.asyncio
@pytest.mark.parametrize("running_container", NIGHTLY_STACKS, indirect=True)
async def test_golden_flow_nightly(api_client, running_container):
    """add -> cognify -> search against the Neo4j (+ Postgres) stack."""
    await run_golden_flow(api_client)


@pytest.mark.nightly
@pytest.mark.asyncio
@pytest.mark.parametrize("running_container", NIGHTLY_STACKS, indirect=True)
async def test_remember_recall_nightly(api_client, running_container):
    """remember -> recall (different dataset) against the Neo4j (+ Postgres) stack."""
    await run_remember_recall_flow(api_client)

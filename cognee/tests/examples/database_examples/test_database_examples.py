"""Mocked tests for examples/database_examples/.

The local/embedded backend (Ladybug) runs under isolated_example_env with no
API key and no network. The external-server backends need a running service:
Neo4j and Postgres/pgvector are deferred to Testcontainers (behind
requires_docker), Neptune needs AWS, and ChromaDB needs its optional extra, so
those are skipped in the base keyless suite.

Part of #3601, on the harness from #3958.
"""

from __future__ import annotations

import pytest

from cognee.tests.utils.example_runner import import_example

pytestmark = pytest.mark.asyncio


async def test_ladybug_example(isolated_example_env):
    module = import_example("examples/database_examples/ladybug_example.py")
    await module.main()


@pytest.mark.skip(reason="Requires the optional chromadb extra; not in the base keyless env.")
async def test_chromadb_example(isolated_example_env):
    module = import_example("examples/database_examples/chromadb_example.py")
    await module.main()


@pytest.mark.skip(
    reason="Requires a running Neo4j server (Testcontainers/requires_docker); deferred follow-up."
)
async def test_neo4j_example(isolated_example_env):
    module = import_example("examples/database_examples/neo4j_example.py")
    await module.main()


@pytest.mark.skip(
    reason="Requires a running Postgres + pgvector server (Testcontainers/requires_docker); deferred follow-up."
)
async def test_pgvector_example(isolated_example_env):
    module = import_example("examples/database_examples/pgvector_example.py")
    await module.main()


@pytest.mark.skip(
    reason="Requires AWS Neptune Analytics (boto3 + AWS credentials); skipped in the keyless suite."
)
async def test_neptune_analytics_example(isolated_example_env):
    module = import_example("examples/database_examples/neptune_analytics_example.py")
    await module.main()

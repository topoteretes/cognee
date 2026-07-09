"""Mocked tests for examples/configurations/.

The local Ladybug configuration and the permissions/tenancy examples run under
isolated_example_env with no API key and no network. The external-database
configurations (Neo4j, Postgres/pgvector, Neptune) need a running service and
are skipped in the base keyless suite (Testcontainers/AWS are the follow-up).

Part of #3601, on the harness from #3958.
"""

from __future__ import annotations

import pytest

from cognee.tests.utils.example_runner import import_example

pytestmark = pytest.mark.asyncio


# Local backend configuration.


async def test_ladybug_graph_database_configuration(isolated_example_env):
    module = import_example(
        "examples/configurations/database_examples/ladybug_graph_database_configuration.py"
    )
    await module.main()


# Permissions / multi-tenancy examples (local).


async def test_data_access_control_example(isolated_example_env):
    module = import_example(
        "examples/configurations/permissions_example/data_access_control_example.py"
    )
    await module.main()


async def test_tenant_role_constraints_example(isolated_example_env):
    module = import_example(
        "examples/configurations/permissions_example/tenant_role_constraints_example.py"
    )
    await module.main()


async def test_tenant_role_setup_example(isolated_example_env):
    module = import_example(
        "examples/configurations/permissions_example/tenant_role_setup_example.py"
    )
    await module.main()


async def test_user_permissions_and_access_control_example(isolated_example_env):
    module = import_example(
        "examples/configurations/permissions_example/user_permissions_and_access_control_example.py"
    )
    await module.main()


# External-database configurations (skipped in the base keyless suite).


@pytest.mark.skip(
    reason="Requires a running Neo4j server (Testcontainers/requires_docker); deferred follow-up."
)
async def test_neo4j_graph_database_configuration(isolated_example_env):
    module = import_example(
        "examples/configurations/database_examples/neo4j_graph_database_configuration.py"
    )
    await module.main()


@pytest.mark.skip(
    reason="Requires a running Postgres + pgvector server (Testcontainers/requires_docker); deferred follow-up."
)
async def test_pgvector_postgres_vector_database_configuration(isolated_example_env):
    module = import_example(
        "examples/configurations/database_examples/pgvector_postgres_vector_database_configuration.py"
    )
    await module.main()


@pytest.mark.skip(
    reason="Requires AWS Neptune Analytics (boto3 + AWS credentials); skipped in the keyless suite."
)
async def test_neptune_analytics_aws_database_configuration(isolated_example_env):
    module = import_example(
        "examples/configurations/database_examples/neptune_analytics_aws_database_configuration.py"
    )
    await module.main()

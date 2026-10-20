"""Connection-string sources: the ALLOW_DATABASE_INGESTION switch, and network
databases that resolve to a server cognee itself uses."""

from types import SimpleNamespace

import pytest

import cognee.infrastructure.databases.cache.config as cache_config
import cognee.infrastructure.databases.graph.config as graph_config
import cognee.infrastructure.databases.relational.config as relational_config
import cognee.infrastructure.databases.vector.config as vector_config
from cognee.exceptions import CogneeValidationError
from cognee.tasks.ingestion.create_dlt_source import (
    ALLOW_DATABASE_INGESTION_ENV,
    _require_foreign_database_server,
    create_dlt_source_from_connection_string,
)

# TEST-NET-3 (RFC 5737): numeric, so resolving it needs no DNS.
FOREIGN = "203.0.113.5"


@pytest.fixture
def servers(monkeypatch):
    """cognee's own databases: relational on localhost:5432, pgvector on
    127.0.0.2:5433 (by URL), session cache on 127.0.0.3 (default port)."""
    monkeypatch.delenv(ALLOW_DATABASE_INGESTION_ENV, raising=False)
    monkeypatch.setattr(
        relational_config,
        "get_relational_config",
        lambda: SimpleNamespace(db_host="localhost", db_port="5432"),
    )
    monkeypatch.setattr(
        vector_config,
        "get_vectordb_config",
        lambda: SimpleNamespace(
            vector_db_host="",
            vector_db_port=1234,
            vector_db_url="postgresql+asyncpg://cognee:cognee@127.0.0.2:5433/cognee_db",
        ),
    )
    monkeypatch.setattr(
        graph_config,
        "get_graph_config",
        lambda: SimpleNamespace(
            graph_database_host="", graph_database_port=123, graph_database_url=""
        ),
    )
    monkeypatch.setattr(
        cache_config,
        "get_cache_config",
        lambda: SimpleNamespace(cache_db_url="postgresql+asyncpg://c:c@127.0.0.3/cache"),
    )


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:p@localhost:5432/other_db",  # the relational server, any database
        "postgresql://u:p@127.0.0.1/other_db",  # same server, default port, other spelling
        "postgres://u:p@127.0.0.2:5433/dataset_db",  # the pgvector server
        "postgresql://u:p@127.0.0.3/anything",  # the session-cache server
    ],
    ids=["relational", "default-port-alias", "pgvector-url", "cache-url"],
)
def test_cognee_database_servers_are_refused(servers, dsn):
    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        create_dlt_source_from_connection_string(dsn)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:p@/db?host=/var/run/postgresql",  # unix socket
        "postgresql://u:p@/db",  # no host: the driver's default socket
        f"postgresql://u:p@{FOREIGN}/db?hostaddr=127.0.0.1",  # address override
        f"postgresql://u:p@{FOREIGN}/db?service=cognee",  # pg_service.conf entry
    ],
    ids=["socket-option", "no-host", "hostaddr", "service"],
)
def test_targets_the_url_does_not_state_are_refused(servers, dsn):
    with pytest.raises(CogneeValidationError, match="must name the server"):
        create_dlt_source_from_connection_string(dsn)


def test_an_unresolvable_host_is_refused(servers):
    with pytest.raises(CogneeValidationError, match="does not resolve"):
        create_dlt_source_from_connection_string("postgresql://u:p@no-such-host.invalid/db")


@pytest.mark.parametrize(
    "dsn",
    [
        f"postgresql://u:p@{FOREIGN}:5432/shop",  # another server
        "postgresql://u:p@127.0.0.1:6543/shop",  # cognee's address, another port
    ],
    ids=["foreign-server", "same-address-other-port"],
)
def test_other_servers_are_allowed(servers, dsn):
    _require_foreign_database_server(dsn)


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE "])
@pytest.mark.parametrize(
    "dsn",
    [f"postgresql://u:p@{FOREIGN}/shop", "sqlite:////tmp/shop.db", "mysql://u:p@db/shop"],
    ids=["postgres", "sqlite", "mysql"],
)
def test_switch_off_refuses_every_connection_string(servers, monkeypatch, value, dsn):
    monkeypatch.setenv(ALLOW_DATABASE_INGESTION_ENV, value)

    with pytest.raises(CogneeValidationError, match="disabled on this server"):
        create_dlt_source_from_connection_string(dsn)

"""Connection-string sources: the ALLOW_DATABASE_INGESTION switch, and network
databases that resolve to a server cognee itself uses."""

from contextlib import nullcontext
from ipaddress import ip_address
from types import SimpleNamespace

import pytest

import cognee.infrastructure.databases.cache.config as cache_config
import cognee.infrastructure.databases.graph.config as graph_config
import cognee.infrastructure.databases.relational.config as relational_config
import cognee.infrastructure.databases.vector.config as vector_config
from cognee.exceptions import CogneeValidationError
from cognee.modules.operations import ORIGIN_API, ORIGIN_CLI, operation_origin_scope
from cognee.tasks.ingestion.create_dlt_source import (
    ALLOW_DATABASE_INGESTION_ENV,
    _require_database_ingestion_allowed,
    _require_foreign_database_server,
    _same_machine,
    create_dlt_source_from_connection_string,
)

# TEST-NET-3 (RFC 5737): numeric, so resolving it needs no DNS.
FOREIGN = "203.0.113.5"


@pytest.fixture
def servers(monkeypatch):
    """cognee's own databases: relational on localhost:5432 as cognee/cognee,
    pgvector on 127.0.0.2:5433 (by URL, as vec/vecpass), session cache on
    127.0.0.3 (default port, as c/c); graph with a user but no password."""
    monkeypatch.delenv(ALLOW_DATABASE_INGESTION_ENV, raising=False)
    monkeypatch.setattr(
        relational_config,
        "get_relational_config",
        lambda: SimpleNamespace(
            db_host="localhost", db_port="5432", db_username="cognee", db_password="cognee"
        ),
    )
    monkeypatch.setattr(
        vector_config,
        "get_vectordb_config",
        lambda: SimpleNamespace(
            vector_db_host="",
            vector_db_port=1234,
            vector_db_username="",
            vector_db_password="",
            vector_db_url="postgresql+asyncpg://vec:vecpass@127.0.0.2:5433/cognee_db",
        ),
    )
    monkeypatch.setattr(
        graph_config,
        "get_graph_config",
        lambda: SimpleNamespace(
            graph_database_host="",
            graph_database_port=123,
            graph_database_url="",
            graph_database_username="neo4j",
            graph_database_password="",
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
        "postgresql://u:p@127.0.0.2:5432/other_db",  # another loopback address
        "postgresql://u:p@0.0.0.0:5432/other_db",  # the unspecified address connects locally
        "postgresql://u:p@[::ffff:127.0.0.1]:5432/other_db",  # IPv4-mapped IPv6
        "postgresql://u:p@[::]:5432/other_db",  # the IPv6 unspecified address
    ],
    ids=["other-loopback", "unspecified", "ipv4-mapped", "ipv6-unspecified"],
)
def test_other_addresses_of_the_same_machine_are_refused(servers, dsn):
    """Resolved addresses are compared as addresses, not strings: every one of
    these reaches the Postgres that cognee's settings call ``localhost``."""
    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        create_dlt_source_from_connection_string(dsn)


@pytest.mark.parametrize(
    "dsn",
    [
        f"postgresql://cognee:cognee@{FOREIGN}:5432/cognee_db",  # relational credentials
        f"postgresql://vec:vecpass@{FOREIGN}/cognee_db",  # pgvector URL credentials
        f"mysql://c:c@{FOREIGN}/cache",  # session-cache URL credentials, any scheme
        "postgresql://cognee:cognee@no-such-host.invalid/cognee_db",  # no address needed
        "postgresql://cognee:cognee@host.docker.internal.invalid:5432/cognee_db",
    ],
    ids=["relational", "vector-url", "cache-url", "unresolvable-host", "other-alias"],
)
def test_cognees_own_credentials_are_refused_on_any_host(servers, dsn):
    """The server's other addresses cannot all be known (the host's LAN address
    behind compose's host.docker.internal, a rebinding name), but every such
    bypass needs cognee's credentials, which are refused before any resolving."""
    with pytest.raises(CogneeValidationError, match="cognee's own database credentials"):
        create_dlt_source_from_connection_string(dsn)


@pytest.mark.parametrize(
    "dsn",
    [
        f"postgresql://cognee:other@{FOREIGN}/shop",  # cognee's user name, another password
        f"postgresql://other:cognee@{FOREIGN}/shop",  # cognee's password, another user
        f"postgresql://neo4j@{FOREIGN}/shop",  # a cognee user configured without a password
        f"postgresql://neo4j:anything@{FOREIGN}/shop",
    ],
    ids=["same-user", "same-password", "passwordless-pair", "passwordless-pair-with-password"],
)
def test_other_credentials_are_allowed(servers, dsn):
    """Matched as a pair, so a server that merely shares a user name or a password
    with cognee's is not refused, and a cognee pair without a password matches nothing."""
    _require_foreign_database_server(dsn)


def test_this_machines_own_address_is_cognees_server(servers):
    """A cognee database on localhost is the same server under the host's own
    interface address; a foreign address is not, loopback or not."""
    own = frozenset({ip_address("192.0.2.10")})

    assert _same_machine("192.0.2.10", "127.0.0.1", own)
    assert _same_machine("::1", "192.0.2.10", own)
    assert _same_machine("::ffff:192.0.2.10", "192.0.2.10", frozenset())
    assert not _same_machine("192.0.2.10", "127.0.0.1", frozenset())
    assert not _same_machine("203.0.113.5", "203.0.113.6", own)
    assert not _same_machine("203.0.113.5", "::1", own)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:p@/db?host=/var/run/postgresql",  # unix socket
        "postgresql://u:p@/db",  # no host: the driver's default socket
        f"postgresql://u:p@{FOREIGN}/db?hostaddr=127.0.0.1",  # address override
        f"postgresql://u:p@{FOREIGN}/db?service=cognee",  # pg_service.conf entry
        f"postgresql://u:p@{FOREIGN}/db?%68ost=localhost",  # percent-encoded option name
        f"postgresql://u@{FOREIGN}/db?passfile=/root/.pgpass",  # server credential file
        f"mysql+pymysql://u@{FOREIGN}/db?read_default_file=/etc/my.cnf",  # same, pymysql
        f"mssql://u:p@{FOREIGN}/db?driver=SQLite3",  # an ODBC driver that reads files
    ],
    ids=[
        "socket-option",
        "no-host",
        "hostaddr",
        "service",
        "encoded-option",
        "passfile",
        "read-default-file",
        "odbc-driver",
    ],
)
def test_targets_the_url_does_not_state_are_refused(servers, dsn):
    with pytest.raises(CogneeValidationError, match="must name the server"):
        create_dlt_source_from_connection_string(dsn)


@pytest.mark.parametrize(
    "dsn",
    [f"postgresql://u:p@{FOREIGN}:abc/db", "postgresql://u:p@[::1/db"],
    ids=["bad-port", "bad-ipv6"],
)
def test_a_malformed_connection_string_is_refused_not_a_crash(servers, dsn):
    with pytest.raises(CogneeValidationError, match="could not be parsed"):
        create_dlt_source_from_connection_string(dsn)


def test_an_unresolvable_host_is_refused(servers):
    with pytest.raises(CogneeValidationError, match="does not resolve"):
        create_dlt_source_from_connection_string("postgresql://u:p@no-such-host.invalid/db")


@pytest.mark.parametrize(
    "dsn",
    [
        f"postgresql://u:p@{FOREIGN}:5432/shop",  # another server
        "postgresql://u:p@127.0.0.1:6543/shop",  # cognee's address, another port
        f"postgresql://u:p@{FOREIGN}/shop?sslmode=require&connect_timeout=5",  # allowed options
    ],
    ids=["foreign-server", "same-address-other-port", "allowed-options"],
)
def test_other_servers_are_allowed(servers, dsn):
    _require_foreign_database_server(dsn)


@pytest.mark.parametrize("value", ["true", "1", "yes"])
@pytest.mark.parametrize("origin", [None, ORIGIN_API], ids=["sdk", "api"])
def test_cognee_servers_stay_refused_with_the_switch_on(servers, monkeypatch, value, origin):
    """The switch opens the feature; it never opens cognee's own storage."""
    monkeypatch.setenv(ALLOW_DATABASE_INGESTION_ENV, value)
    scope = operation_origin_scope(origin) if origin else nullcontext()

    with scope, pytest.raises(CogneeValidationError, match="cognee's own storage"):
        create_dlt_source_from_connection_string("postgresql://u:p@localhost:5432/other_db")


def test_unset_switch_refuses_api_callers_only(servers, monkeypatch):
    """Unset is the default: the SDK (and CLI, MCP) may ingest a database, the
    API server may not, since anyone who can register is one of its callers."""
    dsn = f"postgresql://u:p@{FOREIGN}/shop"

    _require_database_ingestion_allowed()
    with operation_origin_scope(ORIGIN_CLI):
        _require_database_ingestion_allowed()
    with (
        operation_origin_scope(ORIGIN_API),
        pytest.raises(CogneeValidationError, match="not enabled for API callers"),
    ):
        create_dlt_source_from_connection_string(dsn)

    monkeypatch.setenv(ALLOW_DATABASE_INGESTION_ENV, "true")
    with operation_origin_scope(ORIGIN_API):
        _require_database_ingestion_allowed()


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

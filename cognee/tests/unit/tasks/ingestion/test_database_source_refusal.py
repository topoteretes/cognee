"""Connection-string sources: the ALLOW_DATABASE_INGESTION switch, network
databases that resolve to a server cognee itself uses, and the public-servers-only
rule for network callers."""

from contextlib import nullcontext
from ipaddress import ip_address
from types import SimpleNamespace

import pytest

import cognee.infrastructure.databases.cache.config as cache_config
import cognee.infrastructure.databases.graph.config as graph_config
import cognee.infrastructure.databases.relational.config as relational_config
import cognee.infrastructure.databases.vector.config as vector_config
from cognee.exceptions import CogneeValidationError
from cognee.modules.operations import (
    ORIGIN_API,
    ORIGIN_CLI,
    ORIGIN_MCP,
    ORIGIN_SDK,
    operation_origin_scope,
    remote_caller_scope,
)
from cognee.tasks.ingestion import create_dlt_source
from cognee.tasks.ingestion.create_dlt_source import (
    ALLOW_DATABASE_INGESTION_ENV,
    _pin_to_addresses,
    _require_database_ingestion_allowed,
    _require_foreign_database_server,
    _same_machine,
    create_dlt_source_from_connection_string,
)

# TEST-NET-3 (RFC 5737): numeric, so resolving it needs no DNS.
FOREIGN = "203.0.113.5"
# A globally routable address, also numeric. Nothing here connects to it.
PUBLIC = "8.8.8.8"


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
@pytest.mark.parametrize("caller", [nullcontext, remote_caller_scope], ids=["local", "remote"])
def test_cognee_servers_stay_refused_with_the_switch_on(servers, monkeypatch, value, caller):
    """The switch opens the feature; it never opens cognee's own storage."""
    monkeypatch.setenv(ALLOW_DATABASE_INGESTION_ENV, value)

    with caller(), pytest.raises(CogneeValidationError, match="cognee's own storage"):
        create_dlt_source_from_connection_string("postgresql://u:p@localhost:5432/other_db")


@pytest.mark.parametrize("origin", [ORIGIN_SDK, ORIGIN_CLI, ORIGIN_MCP, ORIGIN_API])
def test_unset_switch_refuses_network_callers_only(servers, monkeypatch, origin):
    """Unset is the default: a local caller (SDK, CLI, MCP over stdio) may ingest a
    database, a network caller (an API request, MCP over sse / http) may not, since
    anyone who can reach the server is one of its callers. The origin decides nothing."""
    dsn = f"postgresql://u:p@{FOREIGN}/shop"

    with operation_origin_scope(origin):
        _require_database_ingestion_allowed()
    with (
        operation_origin_scope(origin),
        remote_caller_scope(),
        pytest.raises(CogneeValidationError, match="disabled by default for network callers"),
    ):
        create_dlt_source_from_connection_string(dsn)

    monkeypatch.setenv(ALLOW_DATABASE_INGESTION_ENV, "true")
    with remote_caller_scope():
        _require_database_ingestion_allowed()


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE ", "flase", "ture"])
@pytest.mark.parametrize(
    "dsn",
    [f"postgresql://u:p@{FOREIGN}/shop", "sqlite:////tmp/shop.db", "mysql://u:p@db/shop"],
    ids=["postgres", "sqlite", "mysql"],
)
def test_switch_off_refuses_every_connection_string(servers, monkeypatch, value, dsn):
    monkeypatch.setenv(ALLOW_DATABASE_INGESTION_ENV, value)

    with pytest.raises(CogneeValidationError, match="disabled on this server") as error:
        create_dlt_source_from_connection_string(dsn)

    assert f"ALLOW_DATABASE_INGESTION={value.strip()};" in error.value.message
    assert "Set ALLOW_DATABASE_INGESTION=true" in error.value.message


def _resolve_example_host_to(monkeypatch, addresses):
    """Make ``db.example.com`` resolve to ``addresses``; other hosts resolve as usual."""
    resolve = create_dlt_source._resolved_addresses
    monkeypatch.setattr(
        create_dlt_source,
        "_resolved_addresses",
        lambda host: addresses if host == "db.example.com" else resolve(host),
    )


@pytest.fixture
def network_caller(servers, monkeypatch):
    """A network caller on a server that lets its callers ingest databases."""
    monkeypatch.setenv(ALLOW_DATABASE_INGESTION_ENV, "true")
    with remote_caller_scope():
        yield


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.9",
        "10.0.4.17",
        "172.16.0.5",
        "192.168.1.20",
        "169.254.169.254",
        "100.64.0.1",
        FOREIGN,
        "[::1]",
        "[fd00::5]",
        "[fe80::1]",
        "[::ffff:10.0.4.17]",
        "2130706433",
        "0x7f.0.0.9",
    ],
    ids=[
        "loopback",
        "private-10",
        "private-172",
        "private-192",
        "cloud-metadata",
        "carrier-grade-nat",
        "reserved",
        "ipv6-loopback",
        "ipv6-unique-local",
        "ipv6-link-local",
        "ipv4-mapped-private",
        "decimal-loopback",
        "hex-loopback",
    ],
)
def test_network_callers_are_refused_non_public_hosts(network_caller, host):
    """Port 6543 is not one of cognee's, so only the public-servers rule refuses these.
    The numeric spellings are not IP literals but resolve to loopback."""
    with pytest.raises(CogneeValidationError, match="only databases on public servers"):
        create_dlt_source_from_connection_string(f"postgresql://u:p@{host}:6543/shop")


def test_network_callers_are_refused_a_host_with_any_non_public_address(
    network_caller, monkeypatch
):
    _resolve_example_host_to(monkeypatch, [PUBLIC, "10.0.4.17"])

    with pytest.raises(CogneeValidationError, match="only databases on public servers"):
        _require_foreign_database_server("postgresql://u:p@db.example.com/shop")


def test_network_callers_learn_nothing_from_an_unresolvable_host(network_caller):
    """The same refusal as a private address: no probe of internal DNS names."""
    with pytest.raises(CogneeValidationError, match="only databases on public servers") as error:
        create_dlt_source_from_connection_string("postgresql://u:p@no-such-host.invalid/db")

    assert "no-such-host" not in error.value.message


def test_network_callers_are_admitted_public_hosts(network_caller, monkeypatch):
    assert _require_foreign_database_server(f"postgresql://u:p@{PUBLIC}/shop") == [PUBLIC]

    _resolve_example_host_to(monkeypatch, ["8.8.4.4", PUBLIC])
    assert _require_foreign_database_server("postgresql://u:p@db.example.com/shop") == [
        "8.8.4.4",
        PUBLIC,
    ]


@pytest.mark.parametrize("host", ["127.0.0.9", "10.0.4.17", "192.168.1.20", FOREIGN])
def test_local_callers_keep_non_public_hosts(servers, host):
    """The SDK, CLI and MCP over stdio: the caller already owns the machine."""
    assert _require_foreign_database_server(f"postgresql://u:p@{host}:6543/shop") == [host]


@pytest.mark.parametrize("scheme", ["postgresql", "postgres"])
def test_postgres_is_pinned_by_hostaddr_and_keeps_its_host_name(scheme):
    """Every checked address, in order, so the driver still falls back from one to
    the next; libpq wants the host name once per address."""
    dsn = f"{scheme}://u:p@db.example.com:5432/shop?sslmode=verify-full"

    assert _pin_to_addresses(dsn, [PUBLIC]) == (
        dsn,
        {"connect_args": {"host": "db.example.com", "hostaddr": PUBLIC}},
    )
    assert _pin_to_addresses(dsn, ["2001:4860:4860::8888", PUBLIC]) == (
        dsn,
        {
            "connect_args": {
                "host": "db.example.com,db.example.com",
                "hostaddr": f"2001:4860:4860::8888,{PUBLIC}",
            }
        },
    )


@pytest.mark.parametrize(
    "address, pinned",
    [
        (PUBLIC, f"mysql://u:p%40ss@{PUBLIC}:3306/shop?charset=utf8mb4"),
        (
            "2001:4860:4860::8888",
            "mysql://u:p%40ss@[2001:4860:4860::8888]:3306/shop?charset=utf8mb4",
        ),
    ],
    ids=["ipv4", "ipv6"],
)
def test_other_drivers_are_pinned_by_the_url_host(address, pinned):
    dsn = "mysql://u:p%40ss@db.example.com:3306/shop?charset=utf8mb4"

    assert _pin_to_addresses(dsn, [address, "8.8.4.4"]) == (pinned, {})


def test_the_source_connects_to_the_checked_address(servers, monkeypatch):
    """The name is resolved once: what was checked is what the driver is handed.
    Real dlt and a real engine, stopped at the driver's connect."""
    import psycopg2

    connects = []

    def refuse(*args, **kwargs):
        connects.append(kwargs)
        raise psycopg2.OperationalError("stopped at the driver")

    _resolve_example_host_to(monkeypatch, ["2001:4860:4860::8888", PUBLIC])
    monkeypatch.setattr(psycopg2, "connect", refuse)

    with pytest.raises(CogneeValidationError, match="OperationalError"):
        create_dlt_source_from_connection_string("postgresql://u:p@db.example.com/shop")

    assert connects
    for connect in connects:
        assert connect["host"] == "db.example.com,db.example.com"
        assert connect["hostaddr"] == f"2001:4860:4860::8888,{PUBLIC}"

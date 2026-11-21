import importlib
import sys
import types

# The package re-exports the get_graph_engine function under the module's name.
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)

_ADAPTER_MODULE = "cognee.infrastructure.databases.graph.postgres_demo.adapter"


class _StubAdapter:
    def __init__(self, connection_string, schema=""):
        self.connection_string = connection_string


def _build(provider):
    # __wrapped__ skips the engine cache so each call reaches the provider branch.
    return graph_engine_module._create_graph_engine.__wrapped__(
        provider,
        None,
        graph_database_name="cognee_db",
        graph_database_username="cognee",
        graph_database_password="cognee",
        graph_database_host="localhost",
        graph_database_port="5432",
    )


def test_both_postgres_spellings_log_the_demo_notice_once_per_process(monkeypatch, caplog):
    monkeypatch.setitem(
        sys.modules, _ADAPTER_MODULE, types.SimpleNamespace(PostgresDemoAdapter=_StubAdapter)
    )
    monkeypatch.setattr(graph_engine_module, "_postgres_graph_demo_notice_logged", False)

    with caplog.at_level("WARNING"):
        assert isinstance(_build("postgres"), _StubAdapter)
        assert isinstance(_build("postgres_demo"), _StubAdapter)

    notices = [r.getMessage() for r in caplog.records if "demo graph adapter" in r.getMessage()]
    assert len(notices) == 1
    # The alias never says "demo", so the notice names it and the adapter it resolved to.
    assert "'postgres'" in notices[0] and "postgres_demo" in notices[0]
    assert "not production-ready" in notices[0]

import pytest


@pytest.fixture(autouse=True)
def warm_graph_probe(monkeypatch):
    """Neutralize recall's warm-up short-circuit for API unit tests.

    The guard's relational probe would otherwise hit whatever local DB the
    developer has and short-circuit graph-lane tests that expect
    authorized_search to run. Warm-up tests override this stub per test.
    """
    from cognee.modules.recall.methods import graph_warmup

    async def always_warm(user, dataset_ids):
        return graph_warmup.WarmupProbe(graph_warmup.STATE_WARM, graph_warmup._WARM_COUNT)

    monkeypatch.setattr(graph_warmup, "get_graph_build_status", always_warm)
    graph_warmup.clear_warmup_cache()
    yield
    graph_warmup.clear_warmup_cache()


@pytest.fixture
def stub_dataset_context(monkeypatch):
    """Make entering a dataset's database context a no-op.

    For tests whose datasets are stand-ins (``SimpleNamespace(id=uuid4())``)
    that exist in no database: with access control on, the real context would
    look the dataset up to bind its isolated stores and raise
    ``DatasetNotFoundError``. remember()'s session writes run inside that
    context (so session vectors land in the dataset's store), which is what
    brings these tests to it. Request it with
    ``pytestmark = pytest.mark.usefixtures("stub_dataset_context")``.
    """
    from contextlib import nullcontext

    import cognee.context_global_variables as context_module

    monkeypatch.setattr(
        context_module, "set_database_global_context_variables", lambda *a, **k: nullcontext()
    )


@pytest.fixture
def stub_session_dataset(monkeypatch):
    """Resolve and bind a plain-text ``remember(session_id=...)`` without databases.

    The session path resolves its dataset from the session's binding
    (``_resolve_entry_dataset``: session row lookup, write permission on an
    explicit id) and binds the session before writing. For tests whose users and
    datasets are stand-ins, this resolves to the passed ``dataset_id`` (or a
    fresh id) under the passed name, as an unbound session, and binds nothing.
    Tests of the resolution itself must not request it.
    """
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from uuid import uuid4

    remember_module = importlib.import_module("cognee.api.v1.remember.remember")

    async def resolve(*, session_id, dataset_name, dataset_id, user):
        return user, SimpleNamespace(id=dataset_id or uuid4(), name=dataset_name), False

    monkeypatch.setattr(remember_module, "_resolve_entry_dataset", resolve)
    monkeypatch.setattr(remember_module, "_bind_session_dataset", AsyncMock())

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

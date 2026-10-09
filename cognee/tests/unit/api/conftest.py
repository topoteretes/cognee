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

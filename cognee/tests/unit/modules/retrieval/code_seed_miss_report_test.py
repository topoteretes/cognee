"""A multi-dataset CODE search reports each dataset's miss with its stable code.

Runs the real per-dataset fan-out in cognee/modules/search/methods/search.py and
stubs only its boundaries (dataset context, graph probe, retriever output), the
way the search module's own fan-out tests do.
"""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.modules.retrieval.code_retriever import (
    CodeGraphUnavailableError,
    CodeSeedNotFoundError,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

search_mod = importlib.import_module("cognee.modules.search.methods.search")


def _dataset(name):
    return SimpleNamespace(
        id=uuid5(NAMESPACE_OID, name),
        name=name,
        tenant_id=uuid5(NAMESPACE_OID, "tenant"),
        owner_id=uuid5(NAMESPACE_OID, "owner"),
    )


@pytest.mark.asyncio
async def test_per_dataset_code_misses_carry_their_code_and_param(monkeypatch):
    datasets = [_dataset(name) for name in ("api", "web", "jobs")]
    by_id = {dataset.id: dataset for dataset in datasets}
    outcomes = {
        "api": SearchResultPayload(
            result_object=[{"operation": "flows", "mode": "list", "flows": []}],
            search_type=SearchType.CODE,
            dataset_name="api",
            dataset_id=datasets[0].id,
            dataset_tenant_id=datasets[0].tenant_id,
        ),
        "web": CodeSeedNotFoundError(
            "CODE flow name 'checkout' is ambiguous; use an exact id or repo.",
            code="ambiguous",
            param="name",
        ),
        "jobs": CodeGraphUnavailableError(
            "CODE flows are not available: this dataset's code graph has no flows.",
            code="feature_unavailable",
            param="operation",
        ),
    }
    current = {}

    @asynccontextmanager
    async def dataset_context(dataset_id, *_args, **_kwargs):
        current["name"] = by_id[dataset_id].name
        yield

    class _Engine:
        async def is_empty(self):
            return False

    async def get_graph_engine():
        return _Engine()

    async def get_dataset_data(_dataset_id):
        return ["doc-1"]

    async def get_retriever_output(**kwargs):
        outcome = outcomes[kwargs["dataset"].name]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(search_mod, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "set_database_global_context_variables", dataset_context)
    monkeypatch.setattr(search_mod, "get_graph_engine", get_graph_engine)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.data.methods"),
        "get_dataset_data",
        get_dataset_data,
    )
    monkeypatch.setattr(search_mod, "get_retriever_output", get_retriever_output)

    results = await search_mod.search_in_datasets_context(
        search_datasets=datasets,
        query_type=SearchType.CODE,
        query_text="",
        user=SimpleNamespace(id="u1", tenant_id=None),
    )

    by_name = {payload.dataset_name: payload for payload in results}
    assert by_name["api"].error is None
    web = by_name["web"]
    assert web.completion == {
        "seed_not_found": True,
        "error": web.error,
        "code": "ambiguous",
        "param": "name",
    }
    jobs = by_name["jobs"]
    assert jobs.completion["code"] == "feature_unavailable"
    assert jobs.completion["param"] == "operation"
    assert "has no flows" in jobs.error

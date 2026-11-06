"""The CODE findings operation: stored memories about code, by symbol walk or by repository, with status.

The fixture is the graph the memory join leaves behind: symbols with their
versioned declarations, ObservedDeclaration nodes joined to them by
resolves_to edges carrying the match, and CodeFinding nodes citing the
declarations.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.code_retriever import (
    CodeRetriever,
    CodeSearchValidationError,
    invalidate_code_graph_snapshot_cache,
)
from cognee.tasks.code_graph.extract_code_graph import fact_node_id

REPO = "shop"
CREATE = "shop/orders.create_order"
CHARGE = "shop/payments.charge_card"
VALIDATE = "shop/auth.validate_token"
HELPER = "shop/utils.helper"  # the code a finding was about, gone from the graph
CREATE_ID, CHARGE_ID, VALIDATE_ID = (
    str(fact_node_id(REPO, "symbol", n)) for n in (CREATE, CHARGE, VALIDATE)
)
E_CREATE, E_CHARGE, E_VALIDATE, E_HELPER = "1" * 32, "2" * 32, "3" * 32, "4" * 32
O_CREATE, O_CHARGE, O_HELPER, O_SHARED = "o1", "o2", "o3", "o4"
F_ONE, F_TWO, F_GONE, F_SHARED = "f1", "f2", "f3", "f4"


@pytest.fixture(autouse=True)
def clear_code_graph_snapshot_cache():
    invalidate_code_graph_snapshot_cache(all_entries=True)
    yield
    invalidate_code_graph_snapshot_cache(all_entries=True)


def _symbol(node_id, name, file_path, enola_id, body_hash):
    return (
        node_id,
        {
            "type": "CodeSymbol",
            "kind": "symbol",
            "name": name,
            "file_path": file_path,
            "line": 1,
            "repo": REPO,
            "symbol_kind": "function",
            "enola_id": enola_id,
            "declarations": [
                {"id": enola_id, "file": file_path, "line": 1, "body_hash": body_hash}
            ],
        },
    )


def _observed(node_id, name, enola_id, body_hash, file_path="shop/x.py"):
    return (
        node_id,
        {
            "type": "ObservedDeclaration",
            "name": name,
            "repo": REPO,
            "enola_id": enola_id,
            "file": file_path,
            "line": 1,
            "body_hash": body_hash,
            "decl_hash": None,
            "observed_commit": "c" * 40,
            "extractor_version": "0.4.38",
            "resolution": "exact",
        },
    )


def _finding(node_id, entry_id, text, time):
    return (
        node_id,
        {
            "type": "CodeFinding",
            "name": entry_id,
            "text": text,
            "entry_kind": "qa",
            "entry_id": entry_id,
            "session_id": "review-1",
            "user_id": "reviewer",
            "time": time,
        },
    )


def _graph():
    nodes = [
        (str(fact_node_id(REPO, "repository", REPO)), {"type": "CodeRepository", "name": REPO}),
        _symbol(CREATE_ID, CREATE, "shop/orders.py", E_CREATE, "aaaa"),
        _symbol(CHARGE_ID, CHARGE, "shop/payments.py", E_CHARGE, "bbbb"),
        _symbol(VALIDATE_ID, VALIDATE, "shop/auth.py", E_VALIDATE, "cccc"),
        _observed(O_CREATE, CREATE, E_CREATE, "aaaa"),  # current
        _observed(O_CHARGE, CHARGE, E_CHARGE, "old-"),  # changed
        _observed(O_HELPER, HELPER, E_HELPER, "dddd"),  # gone: joined to nothing
        _observed(O_SHARED, "shop/legacy.init", "5" * 32, "aaaa"),  # ambiguous: two candidates
        _finding(F_ONE, "qa-1", "create_order validates before charging", "2026-01-01T10:00:00"),
        _finding(
            F_TWO, "qa-2", "charge_card retries once; create_order calls it", "2026-01-02T10:00:00"
        ),
        _finding(F_GONE, "qa-3", "helper formats money", "2026-01-03T10:00:00"),
        _finding(F_SHARED, "qa-4", "the shared init body", "2026-01-04T10:00:00"),
    ]
    edges = [
        (CREATE_ID, CHARGE_ID, "calls", {}),
        (VALIDATE_ID, CREATE_ID, "calls", {}),
        (F_ONE, O_CREATE, "cites", {}),
        (F_TWO, O_CHARGE, "cites", {}),
        (F_TWO, O_CREATE, "cites", {}),
        (F_GONE, O_HELPER, "cites", {}),
        (F_SHARED, O_SHARED, "cites", {}),
        (O_CREATE, CREATE_ID, "resolves_to", {"match": "exact"}),
        (O_CHARGE, CHARGE_ID, "resolves_to", {"match": "exact"}),
        (O_SHARED, CREATE_ID, "resolves_to", {"match": "ambiguous"}),
        (O_SHARED, VALIDATE_ID, "resolves_to", {"match": "ambiguous"}),
    ]
    return nodes, edges


async def _run(config, query=""):
    nodes, edges = _graph()
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, edges))
    with patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        return await CodeRetriever(
            config={"operation": "findings", **config}
        ).get_retrieved_objects(query)


def _rows(result):
    return [
        (
            entry["symbol"]["name"] if entry["symbol"] else None,
            entry["symbol"]["depth"] if entry["symbol"] else None,
            entry["finding"]["entry_id"],
            entry["match"],
            entry["status"],
        )
        for entry in result["findings"]
    ]


@pytest.mark.asyncio
async def test_findings_around_a_symbol_come_nearest_first_with_their_status():
    result = await _run({"name": CREATE, "max_depth": 1})

    assert result["seeds"][0]["name"] == CREATE
    assert _rows(result) == [
        (CREATE, 0, "qa-1", "exact", "current"),
        (CREATE, 0, "qa-2", "exact", "current"),
        (CREATE, 0, "qa-4", "ambiguous", "ambiguous"),
        (VALIDATE, 1, "qa-4", "ambiguous", "ambiguous"),
        (CHARGE, 1, "qa-2", "exact", "changed"),
    ]
    assert result["by_status"] == {"current": 2, "changed": 1, "ambiguous": 2, "gone": 0}
    assert result["total"] == 5 and result["has_more"] is False
    entry = result["findings"][0]
    assert entry["finding"]["text"] == "create_order validates before charging"
    assert entry["declaration"]["body_hash"] == "aaaa"
    assert entry["declaration"]["observed_commit"] == "c" * 40
    assert entry["symbol"]["file"] == "shop/orders.py"


@pytest.mark.asyncio
async def test_direction_and_status_narrow_the_walk():
    forward = await _run({"name": CREATE, "direction": "forward"})
    assert {row[0] for row in _rows(forward)} == {CREATE, CHARGE}

    changed = await _run({"name": CREATE, "status": "changed"})
    assert _rows(changed) == [(CHARGE, 1, "qa-2", "exact", "changed")]

    nothing = await _run({"name": CREATE, "status": ["gone"]})
    assert nothing["findings"] == [] and nothing["total"] == 0


@pytest.mark.asyncio
async def test_a_repository_listing_includes_the_findings_whose_code_is_gone():
    result = await _run({"repo": REPO})

    assert result["repo"] == REPO and "seeds" not in result
    assert (None, None, "qa-3", None, "gone") in _rows(result)
    assert result["by_status"]["gone"] == 1
    gone = await _run({"repo": REPO, "status": "gone"})
    assert _rows(gone) == [(None, None, "qa-3", None, "gone")]
    assert gone["findings"][0]["declaration"]["name"] == HELPER


@pytest.mark.asyncio
async def test_paging_and_argument_errors():
    page = await _run({"name": CREATE, "limit": 2, "offset": 1})
    assert [row[2] for row in _rows(page)] == ["qa-2", "qa-4"]
    assert (page["total"], page["has_more"], page["truncated"]) == (5, True, True)

    with pytest.raises(CodeSearchValidationError) as no_scope:
        await _run({})
    assert no_scope.value.error_param == "repo"
    with pytest.raises(CodeSearchValidationError) as bad_status:
        await _run({"name": CREATE, "status": "stale"})
    assert bad_status.value.error_param == "status"


@pytest.mark.asyncio
async def test_capabilities_report_the_findings_the_graph_holds():
    nodes, edges = _graph()
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, edges))
    with patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        result = await CodeRetriever(config={"operation": "capabilities"}).get_retrieved_objects("")

    assert "findings" in result["operations"]
    assert result["features"]["findings"] is True
    assert result["counts"]["findings"] == 4
    assert result["limits"]["findings"] == {
        "limit": {"default": 50, "max": 200},
        "max_depth": {"default": 1, "max": 3},
    }
    # The held-aside memory nodes are not facts: the symbol count is unchanged.
    assert result["counts"]["symbols"] == 3 and result["counts"]["nodes"] == 3

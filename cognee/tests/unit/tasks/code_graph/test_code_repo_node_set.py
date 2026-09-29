"""node_set reaching CODE_REPO-route code nodes (SDK-864).

A call-level node_set already reaches a repo manifest's own Data row and its
documents (classify_documents); these tests cover the code graph half:
extract_code_repo_graph reading the tag off the manifest's external_metadata
and map_facts_to_data_points/extract_code_graph/add_code_graph_edges carrying
it onto every code node, including CodeRepository, correctly across re-syncs.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_data_points,
    add_code_graph_edges,
    extract_code_graph,
    fact_node_id,
    map_facts_to_data_points,
)
from cognee.tasks.code_graph.models import CodeRepository, CodeSymbol

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")
code_graph_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")

REPO = "demo_repo"
SNAPSHOT_ID = "sha256:feedface"

FACTS = [
    {"kind": "symbol", "name": "app/db.Database", "file": "app/db.py", "repo": REPO},
    {
        "kind": "symbol",
        "name": "app/api.handler",
        "file": "app/api.py",
        "repo": REPO,
        "relations": [{"kind": "calls", "target": "app/db.Database"}],
    },
]


def _node_id(kind, name):
    return str(fact_node_id(REPO, kind, name))


REPO_NODE_ID = _node_id("repository", REPO)


def _nodeset_id(name: str) -> str:
    return str(generate_node_id(f"NodeSet:{name}"))


def _write_snapshot(tmp_path, facts, snapshot_id=SNAPSHOT_ID):
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps(fact) for fact in facts) + "\n")
    (tmp_path / "receipt.json").write_text(json.dumps({"snapshot_id": snapshot_id}))
    return tmp_path


def _mock_engine(monkeypatch, nodes=(), edges=(), get_node_return=None):
    engine = AsyncMock()
    engine.get_graph_data.return_value = (list(nodes), list(edges))
    engine.get_node.return_value = get_node_return
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )
    return engine


# ---------------------------------------------------------------------------
# map_facts_to_data_points: tagging + hash behaviour
# ---------------------------------------------------------------------------


def test_every_node_including_repository_carries_the_tag():
    data_points = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=["team-x"])

    assert len(data_points) == 3  # repository + two symbols
    for point in data_points:
        assert point.source_node_set == "team-x"
        assert [ns.name for ns in point.belongs_to_set] == ["team-x"]
        assert point.belongs_to_set[0].id == generate_node_id("NodeSet:team-x")
    assert any(isinstance(point, CodeRepository) for point in data_points)


def test_untagged_call_leaves_belongs_to_set_unset():
    data_points = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}")

    for point in data_points:
        assert point.belongs_to_set is None
        assert point.source_node_set is None


def test_untagged_hash_is_unaffected_by_the_node_set_parameter():
    """Golden check: node_set=None (the default) and node_set=[] both leave
    fact_hash identical to a call that never mentions node_set at all -- an
    untagged repo's hash cannot drift now that the parameter exists."""
    baseline = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}")
    explicit_none = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=None)
    empty = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=[])

    baseline_hashes = {p.name: getattr(p, "fact_hash", None) for p in baseline}
    for other in (explicit_none, empty):
        assert {p.name: getattr(p, "fact_hash", None) for p in other} == baseline_hashes


def test_node_set_changes_the_fact_hash():
    tagged_a = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=["a"])
    tagged_b = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=["b"])
    untagged = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}")

    hashes_a = {p.name: p.fact_hash for p in tagged_a if isinstance(p, CodeSymbol)}
    hashes_b = {p.name: p.fact_hash for p in tagged_b if isinstance(p, CodeSymbol)}
    hashes_untagged = {p.name: p.fact_hash for p in untagged if isinstance(p, CodeSymbol)}

    assert hashes_a != hashes_b
    assert hashes_a != hashes_untagged


def test_node_set_hash_is_order_independent():
    forward = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=["a", "b"])
    backward = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=["b", "a"])

    forward_hashes = {p.name: getattr(p, "fact_hash", None) for p in forward}
    backward_hashes = {p.name: getattr(p, "fact_hash", None) for p in backward}
    assert forward_hashes == backward_hashes


def test_two_repos_different_node_sets_do_not_cross_contaminate(monkeypatch):
    """Two manifests (two calls) of the same dataset, each with its own tag:
    each repo's mapped nodes carry only their own tag."""
    facts_a = [{"kind": "module", "name": "checkout", "repo": "acme/shop"}]
    facts_b = [{"kind": "module", "name": "invoices", "repo": "acme/billing"}]

    points_a = map_facts_to_data_points(facts_a, repo_path="/tmp/shop", node_set=["shop-team"])
    points_b = map_facts_to_data_points(
        facts_b, repo_path="/tmp/billing", node_set=["billing-team"]
    )

    for point in points_a:
        assert [ns.name for ns in point.belongs_to_set] == ["shop-team"]
    for point in points_b:
        assert [ns.name for ns in point.belongs_to_set] == ["billing-team"]


# ---------------------------------------------------------------------------
# get_graph_from_model: belongs_to_set actually produces a NodeSet node + edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_belongs_to_set_produces_nodeset_node_and_edges():
    data_points = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=["team-x"])

    added_nodes: dict = {}
    added_edges: dict = {}
    all_nodes = []
    all_edges = []
    for point in data_points:
        nodes, edges = await get_graph_from_model(
            point, added_nodes=added_nodes, added_edges=added_edges
        )
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    # get_graph_from_model rebuilds graph nodes through copy_model (a same-named
    # but unrelated class), so identity is checked by type name, not isinstance.
    nodeset_nodes = [node for node in all_nodes if type(node).__name__ == "NodeSet"]
    assert len(nodeset_nodes) == 1  # one NodeSet, shared by every node
    assert nodeset_nodes[0].id == generate_node_id("NodeSet:team-x")

    belongs_to_set_edges = {
        (str(source), str(target))
        for source, target, relationship, _props in all_edges
        if relationship == "belongs_to_set"
    }
    expected_targets = {str(generate_node_id("NodeSet:team-x"))}
    assert {target for _source, target in belongs_to_set_edges} == expected_targets
    # Every mapped node (repository + two symbols) has its own belongs_to_set edge.
    assert len(belongs_to_set_edges) == len(data_points)


# ---------------------------------------------------------------------------
# extract_code_graph: snapshot-skip gated on node_set too
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_requires_both_snapshot_id_and_node_set_to_match(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, FACTS)
    engine = _mock_engine(
        monkeypatch,
        get_node_return={"last_snapshot_id": SNAPSHOT_ID, "last_node_set": ["team-x"]},
    )

    data_points = await extract_code_graph(
        repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["team-x"]
    )

    assert data_points == []
    engine.get_node.assert_awaited_once_with(REPO_NODE_ID)


@pytest.mark.asyncio
async def test_previously_untagged_repo_is_fully_reloaded_when_now_tagged(tmp_path, monkeypatch):
    """Same snapshot id, but the repo was loaded without a tag before and now
    carries one: extract_code_graph must not skip, and every mapped node must
    come back tagged (not a partial subset)."""
    _write_snapshot(tmp_path, FACTS)
    _mock_engine(monkeypatch, get_node_return={"last_snapshot_id": SNAPSHOT_ID})

    data_points = await extract_code_graph(
        repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["team-x"]
    )

    assert len(data_points) == 3  # repository + two symbols, none skipped
    for point in data_points:
        assert [ns.name for ns in point.belongs_to_set] == ["team-x"]


@pytest.mark.asyncio
async def test_retagged_repo_is_fully_reloaded_even_with_matching_snapshot_id(
    tmp_path, monkeypatch
):
    _write_snapshot(tmp_path, FACTS)
    _mock_engine(
        monkeypatch,
        get_node_return={"last_snapshot_id": SNAPSHOT_ID, "last_node_set": ["old-team"]},
    )

    data_points = await extract_code_graph(
        repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["new-team"]
    )

    assert len(data_points) == 3
    for point in data_points:
        assert [ns.name for ns in point.belongs_to_set] == ["new-team"]


@pytest.mark.asyncio
async def test_unchanged_untagged_repo_still_skips(tmp_path, monkeypatch):
    """Backstop for the legacy/per-file routes: no node_set given, none stored
    -- skip behaviour is exactly as before this feature existed."""
    _write_snapshot(tmp_path, FACTS)
    engine = _mock_engine(monkeypatch, get_node_return={"last_snapshot_id": SNAPSHOT_ID})

    data_points = await extract_code_graph(repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path)

    assert data_points == []
    engine.get_node.assert_awaited_once_with(REPO_NODE_ID)


# ---------------------------------------------------------------------------
# add_code_graph_edges: stamping and re-tag cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coderepository_keeps_its_tag_after_the_snapshot_stamp(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, FACTS)
    existing_nodes = [(REPO_NODE_ID, {"type": "CodeRepository", "name": REPO})]
    engine = _mock_engine(monkeypatch, existing_nodes, [])

    await add_code_graph_edges(
        ["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["team-x"]
    )

    stamped = engine.add_nodes.await_args.args[0]
    assert len(stamped) == 1
    repository = stamped[0]
    assert repository.last_snapshot_id == SNAPSHOT_ID
    assert repository.last_node_set == ["team-x"]
    assert repository.source_node_set == "team-x"
    # The stamp writes through add_nodes directly (no get_graph_from_model in
    # the way), so belongs_to_set must already be the stored shape: tag names,
    # not NodeSet DataPoints -- otherwise the graph stores full NodeSet dicts.
    assert repository.belongs_to_set == ["team-x"]


@pytest.mark.asyncio
async def test_untagged_repo_is_never_detagged(tmp_path, monkeypatch):
    """No node_set now, none stored before: zero detag calls -- the legacy
    per-file/remember(content_type="code") path pays nothing for this."""
    _write_snapshot(tmp_path, FACTS)
    existing_nodes = [(REPO_NODE_ID, {"type": "CodeRepository", "name": REPO})]
    engine = _mock_engine(monkeypatch, existing_nodes, [])

    await add_code_graph_edges(["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path)

    engine.delete_edge_triples.assert_not_awaited()
    engine.remove_belongs_to_set_tags.assert_not_awaited()


@pytest.mark.asyncio
async def test_retag_strips_old_tag_from_every_surviving_code_node(tmp_path, monkeypatch):
    """node_set change: old tag gone, new tag present, on every surviving
    code node of the repo -- including CodeRepository."""
    _write_snapshot(tmp_path, FACTS)
    handler_id = _node_id("symbol", "app/api.handler")
    database_id = _node_id("symbol", "app/db.Database")
    old_nodeset_id = _nodeset_id("old-team")

    existing_nodes = [
        (
            REPO_NODE_ID,
            {
                "type": "CodeRepository",
                "name": REPO,
                "last_snapshot_id": SNAPSHOT_ID,
                "last_node_set": ["old-team"],
                "belongs_to_set": ["old-team"],
            },
        ),
        (database_id, {"type": "CodeSymbol", "repo": REPO, "belongs_to_set": ["old-team"]}),
        (handler_id, {"type": "CodeSymbol", "repo": REPO, "belongs_to_set": ["old-team"]}),
    ]
    existing_edges = [
        (database_id, REPO_NODE_ID, "part_of", {}),
        (handler_id, REPO_NODE_ID, "part_of", {}),
        (handler_id, database_id, "calls", {}),
        (database_id, old_nodeset_id, "belongs_to_set", {}),
        (handler_id, old_nodeset_id, "belongs_to_set", {}),
        (REPO_NODE_ID, old_nodeset_id, "belongs_to_set", {}),
    ]
    engine = _mock_engine(monkeypatch, existing_nodes, existing_edges)

    await add_code_graph_edges(
        ["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["new-team"]
    )

    deleted_edges = {
        (edge.source_id, edge.target_id, edge.relationship_name)
        for call in engine.delete_edge_triples.await_args_list
        for edge in call.args[0]
    }
    expected_stale_edges = {
        (database_id, old_nodeset_id, "belongs_to_set"),
        (handler_id, old_nodeset_id, "belongs_to_set"),
        (REPO_NODE_ID, old_nodeset_id, "belongs_to_set"),
    }
    assert expected_stale_edges <= deleted_edges

    engine.remove_belongs_to_set_tags.assert_awaited_once()
    tags_call = engine.remove_belongs_to_set_tags.await_args
    assert tags_call.args[0] == ["old-team"]
    assert set(tags_call.kwargs["node_ids"]) == {REPO_NODE_ID, database_id, handler_id}

    stamped = engine.add_nodes.await_args.args[0]
    repository = stamped[0]
    assert repository.last_node_set == ["new-team"]
    # Matches the shape existing_nodes seeded above (a real read returns tag
    # names, not NodeSet DataPoints) -- see test_coderepository_keeps_its_tag.
    assert repository.belongs_to_set == ["new-team"]


@pytest.mark.asyncio
async def test_retag_does_not_touch_names_kept_across_the_change(tmp_path, monkeypatch):
    """Only the DROPPED name is detagged; a name present both before and
    after is left alone."""
    _write_snapshot(tmp_path, FACTS)
    existing_nodes = [
        (
            REPO_NODE_ID,
            {
                "type": "CodeRepository",
                "name": REPO,
                "last_node_set": ["keep", "drop"],
                "belongs_to_set": ["keep", "drop"],
            },
        ),
    ]
    engine = _mock_engine(monkeypatch, existing_nodes, [])

    await add_code_graph_edges(
        ["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["keep"]
    )

    engine.remove_belongs_to_set_tags.assert_awaited_once_with(["drop"], node_ids=[REPO_NODE_ID])


@pytest.mark.asyncio
async def test_retag_is_scoped_to_this_repos_code_nodes_only(tmp_path, monkeypatch):
    """A node of another repo (or a non-code node) that happens to carry the
    same old tag name is never touched by this repo's retag."""
    _write_snapshot(tmp_path, FACTS)
    other_repo_symbol = _node_id("symbol", "elsewhere")
    nodeset_node_id = "11111111-1111-1111-1111-111111111111"

    existing_nodes = [
        (
            REPO_NODE_ID,
            {
                "type": "CodeRepository",
                "name": REPO,
                "last_node_set": ["old-team"],
                "belongs_to_set": ["old-team"],
            },
        ),
        (other_repo_symbol, {"type": "CodeSymbol", "repo": "other_repo"}),
        (nodeset_node_id, {"type": "NodeSet", "name": "old-team"}),
    ]
    engine = _mock_engine(monkeypatch, existing_nodes, [])

    await add_code_graph_edges(
        ["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["new-team"]
    )

    tags_call = engine.remove_belongs_to_set_tags.await_args
    assert tags_call.kwargs["node_ids"] == [REPO_NODE_ID]


@pytest.mark.asyncio
async def test_detag_finds_a_tag_only_the_edge_still_remembers(tmp_path, monkeypatch):
    """Multi-generation loss: the marker and the belongs_to_set PROPERTY both
    already read the CURRENT tag (a later generation's write overwrote the
    property wholesale), but a belongs_to_set EDGE to an older, now-orphaned
    NodeSet still exists (edges are additive-only, a property rewrite never
    touches them). Neither the marker nor the property comparison can see
    this -- only the edge scan can, via the NodeSet node's own name."""
    _write_snapshot(tmp_path, FACTS)
    orphan_nodeset_id = _nodeset_id("orphan")

    existing_nodes = [
        (
            REPO_NODE_ID,
            {
                "type": "CodeRepository",
                "name": REPO,
                "last_node_set": ["current"],
                "belongs_to_set": ["current"],
            },
        ),
        (orphan_nodeset_id, {"type": "NodeSet", "name": "orphan"}),
    ]
    existing_edges = [(REPO_NODE_ID, orphan_nodeset_id, "belongs_to_set", {})]
    engine = _mock_engine(monkeypatch, existing_nodes, existing_edges)

    await add_code_graph_edges(
        ["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["current"]
    )

    # "orphan" is edge-only here: the belongs_to_set PROPERTY already reads
    # ["current"], so there is nothing to strip from the property text (the
    # property comparison is by exact string -- see _detag_stale_code_graph).
    # The stranded EDGE is the actual bug this test guards against; the id
    # based edge comparison finds it regardless of the property.
    engine.remove_belongs_to_set_tags.assert_not_awaited()
    deleted_edges = {
        (edge.source_id, edge.target_id, edge.relationship_name)
        for call in engine.delete_edge_triples.await_args_list
        for edge in call.args[0]
    }
    assert (REPO_NODE_ID, orphan_nodeset_id, "belongs_to_set") in deleted_edges


@pytest.mark.asyncio
async def test_detag_runs_before_the_snapshot_stamp(tmp_path, monkeypatch):
    """The stamp (add_nodes) must only ever record a tag whose old edges and
    property entries have already been stripped -- if the stamp ran first, a
    crash right after it would leave last_node_set pointing at the new tag
    while the old belongs_to_set edges were still live."""
    _write_snapshot(tmp_path, FACTS)
    existing_nodes = [
        (
            REPO_NODE_ID,
            {
                "type": "CodeRepository",
                "name": REPO,
                "last_snapshot_id": SNAPSHOT_ID,
                "last_node_set": ["old-team"],
                "belongs_to_set": ["old-team"],
            },
        ),
    ]
    engine = _mock_engine(monkeypatch, existing_nodes, [])

    await add_code_graph_edges(
        ["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["new-team"]
    )

    call_names = [call[0] for call in engine.mock_calls]
    assert "remove_belongs_to_set_tags" in call_names
    assert "add_nodes" in call_names
    assert call_names.index("remove_belongs_to_set_tags") < call_names.index("add_nodes")


# ---------------------------------------------------------------------------
# Real Ladybug engine: the mocked tests above assert the calls made; these
# confirm the actual stored shape and that the detag call chain succeeds
# against the real adapter (not just a mock that accepts anything).
# ---------------------------------------------------------------------------


def _ladybug_engine(monkeypatch, tmp_path):
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    engine = LadybugAdapter(db_path=str(tmp_path / "g.lbug"))
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )
    return engine


async def _sync(repo_path, snapshot_dir, node_set):
    data_points = await extract_code_graph(
        repo_path=repo_path, snapshot_dir=snapshot_dir, node_set=node_set
    )
    state = await add_code_graph_data_points(data_points)
    await add_code_graph_edges(
        state, repo_path=repo_path, snapshot_dir=snapshot_dir, node_set=node_set
    )


@pytest.mark.asyncio
async def test_real_ladybug_stores_tag_names_and_detags_without_raising(tmp_path, monkeypatch):
    """Against a real Ladybug engine (not a mock that accepts any shape):
    the repo node's belongs_to_set is a list of names after a tagged sync,
    and remove_belongs_to_set_tags (no node_ids -- the forget()/delete path)
    succeeds instead of raising TypeError on an unhashable dict."""
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, FACTS)
    engine = _ladybug_engine(monkeypatch, tmp_path / "graph")
    repo_path = f"/repos/{REPO}"

    await _sync(repo_path, snap, ["team-x"])

    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["belongs_to_set"] == ["team-x"]
    assert repo_node["last_node_set"] == ["team-x"]

    # The forget()/delete cleanup path calls this with no node_ids at all.
    await engine.remove_belongs_to_set_tags(["team-x"])
    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["belongs_to_set"] == []


@pytest.mark.asyncio
async def test_crash_after_step_one_does_not_strand_the_old_tag(tmp_path, monkeypatch):
    """Run 2 crashes inside the detag call, after step 1 (add_code_graph_data_points)
    already rewrote the CodeRepository node. Step 1 must have carried the
    PREVIOUS last_node_set forward, or run 3's retry can never compute which
    tag is stale and the old belongs_to_set edge is stranded forever."""
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, FACTS)
    engine = _ladybug_engine(monkeypatch, tmp_path / "graph")
    repo_path = f"/repos/{REPO}"

    # Run 1: tag with "old-team", completes normally.
    await _sync(repo_path, snap, ["old-team"])
    assert (await engine.get_node(REPO_NODE_ID))["last_node_set"] == ["old-team"]

    # Run 2: retag to "new-team", but the detag call raises once -- simulates
    # a crash after add_code_graph_data_points (step 1) committed its write.
    from cognee.infrastructure.databases.graph.ladybug import adapter as ladybug_adapter_module

    original_remove_tags = ladybug_adapter_module.LadybugAdapter.remove_belongs_to_set_tags

    async def _raise_once(self, *args, **kwargs):
        ladybug_adapter_module.LadybugAdapter.remove_belongs_to_set_tags = original_remove_tags
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(
        ladybug_adapter_module.LadybugAdapter, "remove_belongs_to_set_tags", _raise_once
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        await _sync(repo_path, snap, ["new-team"])

    # Step 1 already ran for run 2 (it always runs before the detag/stamp),
    # so it must have preserved last_node_set instead of wiping it to None.
    assert (await engine.get_node(REPO_NODE_ID))["last_node_set"] == ["old-team"]

    # Run 3: retry. The truly-old tag must still be resolvable and removed.
    await _sync(repo_path, snap, ["new-team"])
    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["last_node_set"] == ["new-team"]
    assert repo_node["belongs_to_set"] == ["new-team"]


# ---------------------------------------------------------------------------
# Marker-only detag was insufficient: a tag written ONLY by a run that never
# reached its own stamp (crashed in the sweep, in this case) never advances
# last_node_set, so a detag that trusted the marker alone could never find
# it. _detag_stale_code_graph now also reads the actual belongs_to_set
# property/edges on this repo's nodes, which the crashed run did write.
# ---------------------------------------------------------------------------


def _crash_once_in(function_name: str) -> None:
    """Patch code_graph_module.<function_name> to raise once, restoring the
    real function before raising so a later call in the same run (or a
    retry) is unaffected."""
    original = getattr(code_graph_module, function_name)

    async def _boom(*args, **kwargs):
        setattr(code_graph_module, function_name, original)
        raise RuntimeError(f"simulated crash in {function_name}")

    setattr(code_graph_module, function_name, _boom)


async def _belongs_to_set_edge_targets(engine, node_id: str) -> list[str]:
    rows = await engine.query(
        "MATCH (a:Node)-[r:EDGE]->(b:Node) WHERE a.id=$i AND "
        "r.relationship_name='belongs_to_set' RETURN b.name",
        {"i": node_id},
    )
    return sorted(row[0] for row in rows)


@pytest.mark.asyncio
async def test_intermediate_crash_does_not_strand_its_own_tag(tmp_path, monkeypatch):
    """A completes, B crashes in the sweep step (after step 1 wrote B's tag,
    before B's own stamp), C runs clean. B's tag never reached the marker,
    so the final state must carry only C -- neither A nor B may survive."""
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, FACTS)
    engine = _ladybug_engine(monkeypatch, tmp_path / "graph")
    repo_path = f"/repos/{REPO}"

    await _sync(repo_path, snap, ["A"])

    _crash_once_in("_sweep_stale_code_graph")
    with pytest.raises(RuntimeError, match="simulated crash"):
        await _sync(repo_path, snap, ["B"])

    await _sync(repo_path, snap, ["C"])

    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["last_node_set"] == ["C"]
    assert repo_node["belongs_to_set"] == ["C"]
    assert await _belongs_to_set_edge_targets(engine, REPO_NODE_ID) == ["C"]


@pytest.mark.asyncio
async def test_first_ever_load_crash_still_gets_detagged_on_retry(tmp_path, monkeypatch):
    """The very first load crashes before ever completing a stamp, so the
    marker stays None throughout -- a marker-only detag has nothing to
    compare against. The tag the crashed run actually wrote must still be
    found (via the property/edges) and stripped on the next clean run."""
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, FACTS)
    engine = _ladybug_engine(monkeypatch, tmp_path / "graph")
    repo_path = f"/repos/{REPO}"

    _crash_once_in("_sweep_stale_code_graph")
    with pytest.raises(RuntimeError, match="simulated crash"):
        await _sync(repo_path, snap, ["A"])
    assert (await engine.get_node(REPO_NODE_ID))["last_node_set"] is None

    await _sync(repo_path, snap, ["B"])

    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["last_node_set"] == ["B"]
    assert repo_node["belongs_to_set"] == ["B"]
    assert await _belongs_to_set_edge_targets(engine, REPO_NODE_ID) == ["B"]


@pytest.mark.asyncio
async def test_untagging_after_a_crash_clears_every_stranded_tag(tmp_path, monkeypatch):
    """A completes, B crashes in the sweep step, then an untagged run
    follows. The untagged run must strip both A (the marker) and B
    (stranded on the property/edges), leaving no tag at all."""
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, FACTS)
    engine = _ladybug_engine(monkeypatch, tmp_path / "graph")
    repo_path = f"/repos/{REPO}"

    await _sync(repo_path, snap, ["A"])

    _crash_once_in("_sweep_stale_code_graph")
    with pytest.raises(RuntimeError, match="simulated crash"):
        await _sync(repo_path, snap, ["B"])

    await _sync(repo_path, snap, None)

    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["last_node_set"] is None
    assert repo_node["belongs_to_set"] is None
    assert await _belongs_to_set_edge_targets(engine, REPO_NODE_ID) == []


@pytest.mark.asyncio
async def test_stored_state_read_failure_does_not_wipe_the_marker(tmp_path, monkeypatch):
    """extract_code_graph's snapshot-skip read (_stored_repository_state) is
    a best-effort optimization; when it raises, step 1 must still carry the
    real last_node_set forward from add_code_graph_data_points's own
    (separate, always-fresh) read, not overwrite it with None."""
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, FACTS)
    engine = _ladybug_engine(monkeypatch, tmp_path / "graph")
    repo_path = f"/repos/{REPO}"

    await _sync(repo_path, snap, ["old-team"])
    assert (await engine.get_node(REPO_NODE_ID))["last_node_set"] == ["old-team"]

    async def _raise(*args, **kwargs):
        raise RuntimeError("simulated read failure")

    monkeypatch.setattr(code_graph_module, "_stored_repository_state", _raise)
    await _sync(repo_path, snap, ["old-team"])

    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["last_node_set"] == ["old-team"]
    assert repo_node["belongs_to_set"] == ["old-team"]


@pytest.mark.asyncio
async def test_detag_finds_a_tag_only_the_property_still_carries(tmp_path, monkeypatch):
    """The marker already matches the current tag and there is no
    belongs_to_set edge left pointing anywhere else, but a code node's
    belongs_to_set PROPERTY still lists an old name. Only the property-scan
    branch of the stale union can see this (neither the marker nor the edge
    scan carries "stale-only" here)."""
    _write_snapshot(tmp_path, FACTS)
    database_id = _node_id("symbol", "app/db.Database")
    handler_id = _node_id("symbol", "app/api.handler")

    existing_nodes = [
        (REPO_NODE_ID, {"type": "CodeRepository", "name": REPO, "last_node_set": ["current"]}),
        (
            database_id,
            {"type": "CodeSymbol", "repo": REPO, "belongs_to_set": ["current", "stale-only"]},
        ),
        (handler_id, {"type": "CodeSymbol", "repo": REPO, "belongs_to_set": ["current"]}),
    ]
    engine = _mock_engine(monkeypatch, existing_nodes, [])

    await add_code_graph_edges(
        ["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["current"]
    )

    tags_call = engine.remove_belongs_to_set_tags.await_args
    assert tags_call.args[0] == ["stale-only"]
    assert set(tags_call.kwargs["node_ids"]) == {REPO_NODE_ID, database_id, handler_id}


@pytest.mark.asyncio
async def test_real_ladybug_case_only_retag_keeps_the_edge(tmp_path, monkeypatch):
    """ "Team-A" -> "team-a" is the SAME NodeSet id (generate_node_id
    lowercases). The retag must not strip the just-written belongs_to_set
    edge -- a raw-string stale comparison would treat "Team-A" as a real
    removal and delete the edge "team-a" also depends on."""
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, FACTS)
    engine = _ladybug_engine(monkeypatch, tmp_path / "graph")
    repo_path = f"/repos/{REPO}"

    await _sync(repo_path, snap, ["Team-A"])
    await _sync(repo_path, snap, ["team-a"])

    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["belongs_to_set"] == ["team-a"]
    assert await _belongs_to_set_edge_targets(engine, REPO_NODE_ID) == ["team-a"]


@pytest.mark.asyncio
async def test_real_ladybug_space_only_retag_keeps_the_edge(tmp_path, monkeypatch):
    """ "team a" -> "team_a" is the SAME NodeSet id (generate_node_id turns
    spaces into underscores). Same invariant as the case-only retag above."""
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, FACTS)
    engine = _ladybug_engine(monkeypatch, tmp_path / "graph")
    repo_path = f"/repos/{REPO}"

    await _sync(repo_path, snap, ["team a"])
    await _sync(repo_path, snap, ["team_a"])

    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["belongs_to_set"] == ["team_a"]
    assert await _belongs_to_set_edge_targets(engine, REPO_NODE_ID) == ["team_a"]


@pytest.mark.asyncio
async def test_respelling_retag_strips_old_spelling_from_property_but_keeps_the_shared_edge(
    tmp_path, monkeypatch
):
    """Neo4j-style backend: add_nodes UNIONS belongs_to_set on write, so a
    respelling-only retag (Team-A -> team-a, same NodeSet id) would keep
    both spellings on the property forever unless the old spelling is
    stripped by EXACT STRING. The id-based comparison used for edges would
    never propose "Team-A" as stale here (same id as "team-a"), which is
    correct -- the edge must survive untouched."""
    _write_snapshot(tmp_path, FACTS)
    existing_nodes = [
        (
            REPO_NODE_ID,
            {
                "type": "CodeRepository",
                "name": REPO,
                "last_node_set": ["Team-A"],
                "belongs_to_set": ["Team-A"],
            },
        ),
    ]
    engine = _mock_engine(monkeypatch, existing_nodes, [])

    await add_code_graph_edges(
        ["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=["team-a"]
    )

    engine.remove_belongs_to_set_tags.assert_awaited_once_with(["Team-A"], node_ids=[REPO_NODE_ID])
    engine.delete_edge_triples.assert_not_awaited()

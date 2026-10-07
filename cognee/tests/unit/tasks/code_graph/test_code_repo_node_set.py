"""node_set reaching CODE_REPO-route code nodes (SDK-864).

A call-level node_set already reaches a repo manifest's own Data row and its
documents (classify_documents); these tests cover the code graph half:
extract_code_repo_graph reading the tag off the manifest's external_metadata
and map_facts_to_data_points/extract_code_graph/add_code_graph_edges carrying
it onto every code node, including CodeRepository, correctly across re-syncs
and after a crashed run.
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

    forward = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=["a", "b"])
    backward = map_facts_to_data_points(FACTS, repo_path=f"/repos/{REPO}", node_set=["b", "a"])
    assert {p.name: getattr(p, "fact_hash", None) for p in forward} == {
        p.name: getattr(p, "fact_hash", None) for p in backward
    }


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
# extract_code_graph: an unchanged snapshot is skipped only when untagged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("node_set", "stored"),
    [
        (["team-x"], {"last_snapshot_id": SNAPSHOT_ID}),
        (["team-x"], {"last_snapshot_id": SNAPSHOT_ID, "belongs_to_set": ["team-x"]}),
        (None, {"last_snapshot_id": SNAPSHOT_ID, "belongs_to_set": ["old-team"]}),
        (None, {"last_snapshot_id": SNAPSHOT_ID, "belongs_to_set": '["old-team"]'}),
    ],
    ids=["now-tagged", "same-tag", "untagged-now", "untagged-now-json-text"],
)
async def test_a_tagged_repo_always_reloads_an_unchanged_snapshot(
    tmp_path, monkeypatch, node_set, stored
):
    """Tagged now, or stored with a tag: never skip, so a tag change or a crashed
    retag is always cleaned up on the next run."""
    _write_snapshot(tmp_path, FACTS)
    _mock_engine(monkeypatch, get_node_return=stored)

    data_points = await extract_code_graph(
        repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path, node_set=node_set
    )

    assert len(data_points) == 3  # repository + two symbols, none skipped


@pytest.mark.asyncio
async def test_unchanged_untagged_repo_still_skips(tmp_path, monkeypatch):
    """No node_set given, none stored: skip behaviour is exactly as before."""
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

    assert engine.add_nodes.await_args.args[0][0].belongs_to_set == ["new-team"]


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

    # The forget()/delete cleanup path calls this with no node_ids at all.
    await engine.remove_belongs_to_set_tags(["team-x"])
    repo_node = await engine.get_node(REPO_NODE_ID)
    assert repo_node["belongs_to_set"] == []


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
    """A completes, B crashes in the sweep step (after B's tag was written,
    before B's stamp), C runs clean: only C may survive, on property and edges."""
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
    assert repo_node["belongs_to_set"] == ["C"]
    assert await _belongs_to_set_edge_targets(engine, REPO_NODE_ID) == ["C"]


@pytest.mark.asyncio
async def test_untagging_after_a_crash_clears_every_stranded_tag(tmp_path, monkeypatch):
    """A completes, B crashes in the sweep step, then an untagged run follows:
    no tag may survive."""
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
    assert repo_node["belongs_to_set"] is None
    assert await _belongs_to_set_edge_targets(engine, REPO_NODE_ID) == []


@pytest.mark.asyncio
async def test_real_ladybug_case_only_retag_keeps_the_edge(tmp_path, monkeypatch):
    """ "Team-A" -> "team-a" is one NodeSet id: the retag must keep the edge."""
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
async def test_respelling_retag_strips_old_spelling_from_property_but_keeps_the_shared_edge(
    tmp_path, monkeypatch
):
    """Neo4j unions belongs_to_set on write, so the old spelling is stripped from
    the property by exact name while the shared edge (same NodeSet id) stays."""
    _write_snapshot(tmp_path, FACTS)
    existing_nodes = [
        (
            REPO_NODE_ID,
            {
                "type": "CodeRepository",
                "name": REPO,
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


def _repo_manifest_item(tmp_path, external_metadata):
    """A CODE_REPO manifest row pointing at an existing directory."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repo_path": str(repo_dir)}))
    return SimpleNamespace(
        id="manifest-1",
        raw_data_location=str(manifest),
        system_metadata={"source": "code_repo"},
        external_metadata=external_metadata,
    )


def _stub_code_graph_tasks(monkeypatch):
    module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")
    extract = AsyncMock(return_value=[])
    monkeypatch.setattr(module, "extract_code_graph", extract)
    monkeypatch.setattr(module, "add_code_graph_data_points", AsyncMock(return_value=[]))
    monkeypatch.setattr(module, "add_code_graph_edges", AsyncMock(return_value=[]))
    return extract


@pytest.mark.asyncio
async def test_repo_route_passes_the_manifest_node_set_to_the_code_graph(tmp_path, monkeypatch):
    from cognee.tasks.code_graph.code_repo import extract_code_repo_graph

    extract = _stub_code_graph_tasks(monkeypatch)
    item = _repo_manifest_item(tmp_path, {"node_set": ["team-a"]})

    await extract_code_repo_graph([item])

    assert extract.await_args.kwargs["node_set"] == ["team-a"]


@pytest.mark.asyncio
async def test_repo_route_rejects_a_malformed_manifest_node_set(tmp_path, monkeypatch):
    """Same rule as classify_documents: a bad tag raises, never loads untagged."""
    from cognee.modules.engine.models.node_set import InvalidNodeSetError
    from cognee.tasks.code_graph.code_repo import extract_code_repo_graph

    extract = _stub_code_graph_tasks(monkeypatch)
    item = _repo_manifest_item(tmp_path, {"node_set": "team-a"})

    with pytest.raises(InvalidNodeSetError):
        await extract_code_repo_graph([item])
    extract.assert_not_awaited()

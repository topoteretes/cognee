"""Observed declarations resolve to symbols by id, follow unambiguous moves, and stay explicit otherwise."""

import json
from unittest.mock import AsyncMock

import pytest

from cognee.tasks.code_graph.extract_code_graph import fact_node_id
from cognee.tasks.code_memory import (
    RESOLVES_TO_RELATIONSHIP,
    CodeSymbolIndex,
    link_observed_declarations,
    observed_declaration,
    observed_declaration_status,
    resolve_observed_declaration,
)

REPO = "httpx"
SEND = "httpx/_client.Client.send"
SEND_ID = "3dd612bf822d62516cc184abebddf52b"
SEND_NODE = str(fact_node_id(REPO, "symbol", SEND))

# The reference a reviewer stored (a real enola 0.4.38 stamp of httpx).
REF = {
    "repo": REPO,
    "enola_id": SEND_ID,
    "name": SEND,
    "file": "httpx/_client.py",
    "line": 646,
    "body_hash": "2d5be8556574e8c3",
    "observed_commit": "a4463d044f370aef186f3c64bd79e236e4df9e2a",
    "extractor_version": "0.4.38",
}


def _symbol(name, declarations, repo=REPO, **extra):
    node_id = str(fact_node_id(repo, "symbol", name))
    properties = {"type": "CodeSymbol", "name": name, "repo": repo, **extra}
    if declarations is not None:
        properties["declarations"] = declarations
    return node_id, properties


def _declaration(enola_id, body_hash, file="httpx/_client.py"):
    return {"id": enola_id, "file": file, "body_hash": body_hash, "line": 1, "end_line": 2}


def _index(*nodes):
    return CodeSymbolIndex.from_nodes(nodes, {REPO})


def test_the_declaration_id_resolves_exactly_even_when_the_body_changed():
    index = _index(_symbol(SEND, [_declaration(SEND_ID, "ffffffffffffffff")]))
    matches = resolve_observed_declaration(REF, index)
    assert matches == [(SEND_NODE, "exact")]
    assert observed_declaration_status(REF, matches, index) == "changed"


def test_an_unchanged_body_is_current():
    index = _index(_symbol(SEND, [_declaration(SEND_ID, REF["body_hash"])]))
    matches = resolve_observed_declaration(REF, index)
    assert observed_declaration_status(REF, matches, index) == "current"


def test_a_unique_body_under_another_name_is_followed_as_a_move():
    moved_to = "httpx/_client._send"
    index = _index(
        _symbol(moved_to, [_declaration("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", REF["body_hash"])])
    )
    matches = resolve_observed_declaration(REF, index)
    assert matches == [(str(fact_node_id(REPO, "symbol", moved_to)), "moved")]
    assert observed_declaration_status(REF, matches, index) == "current"


def test_a_rekeyed_id_over_the_same_name_and_file_is_exact():
    index = _index(
        _symbol(SEND, [_declaration("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", REF["body_hash"])])
    )
    assert resolve_observed_declaration(REF, index) == [(SEND_NODE, "exact")]


def test_a_body_shared_by_several_symbols_is_ambiguous_and_names_every_candidate():
    index = _index(
        _symbol(
            "httpx/_exceptions.CookieConflict.__init__", [_declaration("c" * 32, REF["body_hash"])]
        ),
        _symbol(
            "httpx/_exceptions.InvalidURL.__init__", [_declaration("d" * 32, REF["body_hash"])]
        ),
    )
    matches = resolve_observed_declaration(REF, index)
    assert sorted(match for _node, match in matches) == ["ambiguous", "ambiguous"]
    assert len({node for node, _match in matches}) == 2
    assert observed_declaration_status(REF, matches, index) == "ambiguous"


def test_nothing_with_the_id_or_the_body_is_gone():
    index = _index(_symbol(SEND, [_declaration("e" * 32, "0000000000000000")]))
    assert resolve_observed_declaration(REF, index) == []
    assert observed_declaration_status(REF, [], index) == "gone"


def test_a_symbol_without_versioned_declarations_resolves_by_its_own_enola_id():
    # A graph from a snapshot without body hashes (enola < 0.4.37) keeps only
    # the node's enola_id; the status cannot say current, only changed.
    index = _index(_symbol(SEND, None, enola_id=SEND_ID, file_path="httpx/_client.py"))
    matches = resolve_observed_declaration(REF, index)
    assert matches == [(SEND_NODE, "exact")]
    assert observed_declaration_status(REF, matches, index) == "changed"


def test_declarations_stored_as_json_text_are_read():
    # Neo4j stores a list property as ``<key>_json`` text.
    node_id, properties = _symbol(SEND, None)
    properties["declarations_json"] = json.dumps([_declaration(SEND_ID, REF["body_hash"])])
    index = _index((node_id, properties))
    assert resolve_observed_declaration(REF, index) == [(SEND_NODE, "exact")]


def test_symbols_of_other_repositories_are_not_candidates():
    index = _index(_symbol(SEND, [_declaration(SEND_ID, REF["body_hash"])], repo="fork"))
    assert resolve_observed_declaration(REF, index) == []


@pytest.mark.asyncio
async def test_the_join_is_an_edge_diff_that_replaces_a_changed_match():
    observed = observed_declaration(REF)
    observed_id = str(observed.id)
    moved_to = "httpx/_client._send"
    moved_node = str(fact_node_id(REPO, "symbol", moved_to))
    nodes = [
        (observed_id, {**observed.model_dump(exclude={"id"}), "type": "ObservedDeclaration"}),
        _symbol(moved_to, [_declaration("a" * 32, REF["body_hash"])]),
    ]
    # The graph still holds the edge to the old node, where the symbol was.
    stale_edge = (observed_id, SEND_NODE, RESOLVES_TO_RELATIONSHIP, {"match": "exact"})
    engine = AsyncMock()
    engine.get_filtered_graph_data.return_value = (nodes, [stale_edge])

    counts = await link_observed_declarations(engine, {REPO}, existing_nodes=nodes)

    assert counts == {"exact": 0, "moved": 1, "ambiguous": 0, "unmatched": 0}
    (deleted,) = engine.delete_edge_triples.call_args.args[0]
    assert (deleted.source_id, deleted.target_id) == (observed_id, SEND_NODE)
    ((written,),) = engine.add_edges.call_args.args
    assert written[:3] == (observed_id, moved_node, RESOLVES_TO_RELATIONSHIP)
    assert written[3]["match"] == "moved"


@pytest.mark.asyncio
async def test_an_unchanged_join_writes_nothing():
    observed = observed_declaration(REF)
    observed_id = str(observed.id)
    nodes = [
        (observed_id, {**observed.model_dump(exclude={"id"}), "type": "ObservedDeclaration"}),
        _symbol(SEND, [_declaration(SEND_ID, REF["body_hash"])]),
    ]
    engine = AsyncMock()
    engine.get_filtered_graph_data.return_value = (
        nodes,
        [(observed_id, SEND_NODE, RESOLVES_TO_RELATIONSHIP, {"match": "exact"})],
    )

    counts = await link_observed_declarations(engine, {REPO}, existing_nodes=nodes)

    assert counts["exact"] == 1
    engine.add_edges.assert_not_called()
    engine.delete_edge_triples.assert_not_called()


@pytest.mark.asyncio
async def test_a_repository_no_memory_names_is_not_read():
    engine = AsyncMock()
    nodes = [_symbol(SEND, [_declaration(SEND_ID, REF["body_hash"])])]

    assert await link_observed_declarations(engine, {REPO}, existing_nodes=nodes) == {}

    engine.get_filtered_graph_data.assert_not_called()

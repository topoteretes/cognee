"""Unit tests for deterministic (rule-based) dedup — issue #3627, Approach A.

Deterministic and pure-Python: no LLM, no database, no network. These import the
REAL resolver from cognee.modules.graph.utils and run it on real DataPoint nodes,
so the shipped code — not a copy — is under test.
"""

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import resolve_deterministic_duplicates
from cognee.modules.graph.utils.deterministic_dedup import MERGED_INTO_RELATIONSHIP


class _Entity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class _OtherNamed(DataPoint):
    """A different node type carrying a name, for type-gating tests."""

    name: str
    metadata: dict = {"index_fields": ["name"]}


def test_empty_and_single_node_return_empty():
    assert resolve_deterministic_duplicates([]) == []
    assert resolve_deterministic_duplicates([_Entity(name="Solo")]) == []


def test_case_and_punctuation_variants_are_linked():
    a = _Entity(name="Apple Inc.")
    b = _Entity(name="apple inc")

    merge_edges = resolve_deterministic_duplicates([a, b])

    assert len(merge_edges) == 1
    source, target, relationship_name, properties = merge_edges[0]
    assert relationship_name == MERGED_INTO_RELATIONSHIP
    assert {source, target} == {a.id, b.id}
    assert properties["resolution"] == "deterministic_key"
    assert properties["canonical_key"] == "apple_inc"


def test_alias_variants_collapse_to_one_canonical():
    """USA / U.S.A. / United States all map to the same alias key (default map)."""
    usa = _Entity(name="USA")
    usa_dotted = _Entity(name="U.S.A.")
    full = _Entity(name="United States")

    merge_edges = resolve_deterministic_duplicates([usa, usa_dotted, full])

    # Two duplicates -> two merged_into edges, all pointing to one canonical.
    assert len(merge_edges) == 2
    assert {rel for _, _, rel, _ in merge_edges} == {MERGED_INTO_RELATIONSHIP}
    canonical_ids = {target for _, target, _, _ in merge_edges}
    assert canonical_ids == {full.id}  # longest original name wins
    assert {props["canonical_key"] for *_, props in merge_edges} == {"united_states"}


def test_distinct_names_are_not_linked():
    a = _Entity(name="Apple")
    b = _Entity(name="Google")
    assert resolve_deterministic_duplicates([a, b]) == []


def test_type_gating_prevents_cross_type_merge():
    entity = _Entity(name="Mercury")
    other = _OtherNamed(name="Mercury")  # same canonical key, different node type
    assert resolve_deterministic_duplicates([entity, other]) == []


def test_nodes_without_name_are_ignored():
    class _NoName(DataPoint):
        text: str
        metadata: dict = {"index_fields": ["text"]}

    assert resolve_deterministic_duplicates([_NoName(text="anything")]) == []


def test_same_id_nodes_are_not_self_linked():
    """Identical ids are already collapsed upstream; never emit a self-merge."""
    shared_id = _Entity(name="Apple Inc.").id
    a = _Entity(id=shared_id, name="Apple Inc.")
    b = _Entity(id=shared_id, name="apple inc")
    assert resolve_deterministic_duplicates([a, b]) == []


def test_input_nodes_are_not_mutated():
    a = _Entity(name="Apple Inc.")
    b = _Entity(name="apple inc")
    before = [(n.id, n.name) for n in (a, b)]

    resolve_deterministic_duplicates([a, b])

    assert [(n.id, n.name) for n in (a, b)] == before  # names left intact

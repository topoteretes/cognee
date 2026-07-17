"""Deterministic rule-based dedup + merge (Approach A). No LLM, no graph I/O."""

from cognee.modules.engine.models.Entity import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.modules.graph.utils.canonicalize import canonicalize, canonical_block_key
from cognee.modules.graph.utils.deduplicate_nodes_and_edges import (
    deduplicate_nodes_and_edges,
)


def _entity(name, description="d"):
    et = EntityType(name="Concept", description="c")
    return Entity(name=name, is_a=et, description=description)


class TestCanonicalization:
    def test_case_and_whitespace(self):
        assert canonicalize("  Hello   World  ") == "hello world"

    def test_unicode_nfkc_lookalike(self):
        # U+0139 (Ĺ) NFKC-folds; compatibility forms collapse deterministically.
        assert canonicalize("ﬁle") == "file"  # U+FB01 ligature -> "fi"

    def test_punctuation_and_hyphen(self):
        assert canonicalize("Natural-Language, Processing.") == "natural language processing"

    def test_alias_resolution(self):
        assert canonicalize("NLP") == "natural language processing"
        assert canonicalize("IBM") == canonicalize("International Business Machines")


class TestBlockingKey:
    def test_variants_share_block(self):
        assert canonical_block_key(_entity("NLP")) == canonical_block_key(
            _entity("Natural Language Processing")
        )

    def test_distinct_entities_separate_blocks(self):
        assert canonical_block_key(_entity("Python")) != canonical_block_key(_entity("Java"))

    def test_same_name_different_type_separate(self):
        et = EntityType(name="Concept", description="c")
        assert canonical_block_key(et) != canonical_block_key(_entity("Concept"))


class TestMerge:
    def test_variants_merge_to_one(self):
        variants = [
            "NLP",
            "nlp",
            "Natural Language Processing",
            "Natural-Language Processing",
        ]
        nodes, _ = deduplicate_nodes_and_edges([_entity(v) for v in variants], [])
        assert len(nodes) == 1

    def test_distinct_entities_stay_separate(self):
        nodes, _ = deduplicate_nodes_and_edges([_entity("Python"), _entity("Java")], [])
        assert len(nodes) == 2

    def test_provenance_recorded(self):
        nodes, _ = deduplicate_nodes_and_edges(
            [_entity("NLP"), _entity("Natural Language Processing")], []
        )
        assert nodes[0].metadata.get("merged_from")
        assert nodes[0].metadata.get("canonical_version") == 1

    def test_non_null_wins(self):
        # An empty field on the survivor is filled from a peer (non_null_wins).
        a = _entity("NLP", description="")
        b = _entity("nlp", description="filled in")
        nodes, _ = deduplicate_nodes_and_edges([a, b], [])
        assert nodes[0].description == "filled in"

    def test_list_field_union(self):
        a = _entity("NLP")
        a.relations = [("uses", "x")]
        b = _entity("nlp")
        b.relations = [("uses", "y")]
        nodes, _ = deduplicate_nodes_and_edges([a, b], [])
        assert ("uses", "x") in nodes[0].relations
        assert ("uses", "y") in nodes[0].relations

    def test_deterministic_survivor_id(self):
        variants = ["NLP", "nlp", "Natural Language Processing"]
        run1, _ = deduplicate_nodes_and_edges([_entity(v) for v in variants], [])
        run2, _ = deduplicate_nodes_and_edges([_entity(v) for v in reversed(variants)], [])
        assert run1[0].id == run2[0].id  # order-independent

    def test_idempotent(self):
        variants = ["NLP", "Natural Language Processing"]
        once, _ = deduplicate_nodes_and_edges([_entity(v) for v in variants], [])
        twice, _ = deduplicate_nodes_and_edges(once, [])
        assert len(twice) == 1
        assert twice[0].id == once[0].id


class TestEdgeRemap:
    def test_edges_rewired_to_survivor(self):
        a, b = _entity("NLP"), _entity("Natural Language Processing")
        other = _entity("Linguistics")
        # edge from the soon-to-be-merged-away node onto `other`
        edges = [(b.id, other.id, "related_to", {})]
        nodes, out_edges = deduplicate_nodes_and_edges([a, b, other], edges)
        surviving_ids = {str(n.id) for n in nodes}
        # every edge endpoint must reference a surviving node
        for e in out_edges:
            assert str(e[0]) in surviving_ids
            assert str(e[1]) in surviving_ids

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.utils.query_state import QueryState


def make_triplet(source="node-a", relationship="likes", target="node-b"):
    """Build a triplet the way graph projection does: fresh instances each call."""
    source_node = Node(source, {"name": source, "type": "Entity"})
    target_node = Node(target, {"name": target, "type": "Entity"})
    return Edge(source_node, target_node, attributes={"relationship_name": relationship})


def test_merge_triplets_dedupes_re_retrieved_instances():
    """The same logical triplet must merge to one entry across rounds.

    Regression test: merge_triplets deduplicated by id(), but every retrieval
    round re-projects the graph into fresh Edge instances, so a re-retrieved
    triplet never matched and duplicates accumulated every round.
    """
    state = QueryState([make_triplet()], "ctx")

    state.merge_triplets([make_triplet()])

    assert len(state.triplets) == 1


def test_merge_triplets_keeps_distinct_relationships():
    state = QueryState([make_triplet(relationship="works_at")], "ctx")

    state.merge_triplets([make_triplet(relationship="founded")])

    assert len(state.triplets) == 2


def test_merge_triplets_keeps_distinct_node_pairs():
    state = QueryState([make_triplet()], "ctx")

    state.merge_triplets([make_triplet(target="node-c")])

    assert len(state.triplets) == 2


def test_convergence_triggers_when_round_adds_nothing_new():
    """Regression test: with id()-based dedupe, a round returning the same
    logical triplets still grew the list, so `done` never flipped and context
    extension always ran the maximum number of rounds."""
    state = QueryState([make_triplet()], "ctx")
    prev_size = len(state.triplets)

    state.merge_triplets([make_triplet()])
    state.check_convergence(prev_size)

    assert state.done


def test_no_convergence_when_new_triplet_added():
    state = QueryState([make_triplet()], "ctx")
    prev_size = len(state.triplets)

    state.merge_triplets([make_triplet(relationship="founded")])
    state.check_convergence(prev_size)

    assert not state.done

"""Ordering contract for combining two retrievals of the same shape."""

from unittest.mock import MagicMock

from cognee.modules.retrieval.utils.merge_results import (
    conversational_reserve,
    edge_identity,
    merge_ranked,
)


def result(identifier, text, *, truthy=True):
    item = MagicMock()
    item.id = identifier
    item.payload = {"id": identifier, "text": text}
    if not truthy:
        item.__bool__.return_value = False
    return item


def edge(edge_id, *, relationship="knows"):
    item = MagicMock()
    item.attributes = {"edge_object_id": edge_id, "relationship_name": relationship}
    item.node1.id = "source"
    item.node2.id = "target"
    item.directed = True
    return item


def test_items_in_both_lanes_rank_first_and_keep_the_primary_object():
    primary_a = result("a", "primary a")
    primary_b = result("b", "primary b", truthy=False)
    secondary_b = result("b", "secondary b")
    secondary_c = result("c", "secondary c")

    merged = merge_ranked([primary_a, primary_b], [secondary_b, secondary_c], limit=3)

    # "b" was found by both lanes, so it outranks primary-only "a"; secondary-only last.
    assert merged == [primary_b, primary_a, secondary_c]


def test_primary_only_outranks_secondary_only_regardless_of_rank():
    primary = [result(f"p{index}", "p") for index in range(3)]
    secondary = [result("s0", "s")]

    assert merge_ranked(primary, secondary) == [*primary, secondary[0]]


def test_one_lane_preserves_order_and_cap():
    items = [result("a", "a"), result("b", "b"), result("c", "c")]

    assert merge_ranked(items, None, limit=2) == items[:2]
    assert merge_ranked(None, items) == items


def test_results_without_an_identity_are_never_merged_away():
    anonymous = [result(None, "one"), result(None, "two")]

    assert merge_ranked(anonymous, anonymous) == anonymous + anonymous


def test_repeated_item_inside_one_lane_keeps_its_first_rank():
    first = result("a", "first")
    duplicate = result("a", "duplicate")

    assert merge_ranked([first, duplicate], None) == [first]


def test_edges_merge_by_object_id():
    primary = edge("edge-1")

    assert merge_ranked([primary], [edge("edge-1")], identity=edge_identity) == [primary]


def test_edges_without_an_object_id_merge_by_relationship():
    primary = edge(None)

    assert merge_ranked([primary], [edge(None)], identity=edge_identity) == [primary]


def test_unrelated_edges_are_both_kept():
    primary = edge("edge-1")
    secondary = edge("edge-2")

    merged = merge_ranked([primary], [secondary], identity=edge_identity)

    assert merged == [primary, secondary]


def test_full_lanes_reserve_slots_for_conversational_only_items():
    """Both lanes return exactly top_k with one shared item; reserve keeps ctx-only hits."""
    limit = 5
    primary = [result(f"raw{index}", "raw") for index in range(limit)]
    secondary = [
        result("ctx0", "ctx"),
        result("ctx1", "ctx"),
        result("raw3", "shared"),
        result("ctx2", "ctx"),
        result("ctx3", "ctx"),
    ]

    merged = merge_ranked(
        primary,
        secondary,
        limit=limit,
        secondary_reserve=conversational_reserve(limit),
    )

    # One reserved slot at limit=5, so the lowest-ranked raw item yields to "ctx0".
    assert [item.id for item in merged] == ["raw3", "raw0", "raw1", "raw2", "ctx0"]


def test_empty_secondary_and_zero_reserve_keep_primary_cap():
    primary = [result(f"raw{index}", "raw") for index in range(5)]
    secondary = [result(f"ctx{index}", "ctx") for index in range(5)]

    assert (
        merge_ranked(primary, None, limit=5, secondary_reserve=conversational_reserve(5)) == primary
    )
    assert (
        merge_ranked(primary, [], limit=5, secondary_reserve=conversational_reserve(5)) == primary
    )
    assert merge_ranked(primary, secondary, limit=5, secondary_reserve=0) == primary

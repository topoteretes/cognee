from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cognee.tasks.memify.global_context_index.build as build_module
import cognee.tasks.memify.global_context_index.update as update_module
from cognee.tasks.memify.global_context_index.bucketing.divisive.graph_distance import (
    build_graph_pole_a_fn,
    build_graph_similarity_fn,
)
from cognee.tasks.memify.global_context_index.bucketing.divisive.placement import (
    build_divisive_buckets_for_level,
)
from cognee.tasks.memify.global_context_index.bucketing.divisive.split import divisive_split
from cognee.tasks.memify.global_context_index.bucketing.divisive.vector_distance import (
    build_vector_pole_a_fn,
    build_vector_similarity_fn,
    embed_items_for_divisive_split,
)
from cognee.tasks.memify.global_context_index.bucketing.graph.placement import (
    place_graph_summaries_incrementally,
)
from cognee.tasks.memify.global_context_index.bucketing.graph.scoring import (
    cosine_distance,
    weighted_jaccard,
)
from cognee.tasks.memify.global_context_index.bucketing.vector.placement import (
    assign_items_to_buckets,
)
from cognee.tasks.memify.global_context_index.build import BuildOptions, place_items_for_level
from cognee.tasks.memify.global_context_index.ids import create_bucket_id
from cognee.tasks.memify.global_context_index.models import SummaryNode
from cognee.tasks.memify.global_context_index.update import (
    build_and_persist_context_index,
    validate_global_context_index_config,
)


def _summary(summary_id: str, text: str | None = None) -> SummaryNode:
    return SummaryNode(
        id=summary_id, text=text if text is not None else f"{summary_id} text", type="TextSummary"
    )


def _bucket(bucket_id: str, child_ids: set[str], graph_bucket_entity_ids) -> SummaryNode:
    return SummaryNode(
        id=bucket_id,
        text=f"{bucket_id} text",
        type="GlobalContextSummary",
        level=0,
        child_ids=child_ids,
        graph_bucket_entity_ids=graph_bucket_entity_ids,
    )


def _fake_vector_engine(vectors_by_text: dict[str, list[float]]):
    async def embed_text(texts):
        return [vectors_by_text[text] for text in texts]

    return SimpleNamespace(embedding_engine=SimpleNamespace(embed_text=embed_text))


def _make_similarity(pairs: dict[frozenset, float]):
    def similarity_fn(left_id: str, right_id: str) -> float:
        if left_id == right_id:
            return 1.0
        return pairs[frozenset({left_id, right_id})]

    return similarity_fn


# ---------------------------------------------------------------------------
# split.py
# ---------------------------------------------------------------------------


def test_divisive_split_returns_single_leaf_when_group_fits_max_bucket_size():
    buckets = divisive_split(
        ["c", "a", "b"], lambda a, b: 0.0, lambda ids: ids[0], max_bucket_size=5
    )

    assert buckets == [["a", "b", "c"]]


def test_divisive_split_recurses_until_every_leaf_fits_max_bucket_size():
    groups = {"a1", "a2", "a3", "a4"}, {"b1", "b2", "b3", "b4"}
    membership = {item_id: group_idx for group_idx, group in enumerate(groups) for item_id in group}

    def similarity_fn(a, b):
        return 1.0 if membership[a] == membership[b] else 0.0

    def pole_a_fn(ids):
        return sorted(ids)[0]

    buckets = divisive_split(
        ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"],
        similarity_fn,
        pole_a_fn,
        max_bucket_size=2,
    )

    assert buckets == [["a1", "a2"], ["a3", "a4"], ["b1", "b2"], ["b3", "b4"]]


def test_divisive_split_pole_b_tie_broken_by_ascending_id():
    pairs = {
        frozenset({"seed", "x1"}): 0.5,
        frozenset({"seed", "x2"}): 0.5,
        frozenset({"seed", "x3"}): 0.9,
        frozenset({"x1", "x2"}): 0.5,
        frozenset({"x1", "x3"}): 0.1,
        frozenset({"x2", "x3"}): 0.1,
    }
    similarity_fn = _make_similarity(pairs)

    buckets = divisive_split(
        ["seed", "x1", "x2", "x3"], similarity_fn, lambda ids: "seed", max_bucket_size=3
    )

    # pole B is chosen as "x1" (first ascending-id item tied for lowest score
    # against "seed"), not "x2" -- so "x1" ends up isolated, "x2" stays with seed.
    assert buckets == [["seed", "x2", "x3"], ["x1"]]


def test_divisive_split_side_assignment_tie_goes_to_pole_a():
    pairs = {
        frozenset({"p", "q"}): 0.5,
        frozenset({"p", "m"}): 0.3,
        frozenset({"q", "m"}): 0.5,
    }
    similarity_fn = _make_similarity(pairs)

    buckets = divisive_split(["p", "q", "m"], similarity_fn, lambda ids: "p", max_bucket_size=2)

    # "q" ties exactly (0.5/0.5) between pole A ("p") and pole B ("m") -> stays with "p".
    assert buckets == [["p", "q"], ["m"]]


def test_divisive_split_anti_stall_guard_triggers_on_constructed_all_one_side_input():
    buckets = divisive_split(
        ["a", "b", "c", "d"], lambda a, b: 1.0, lambda ids: ids[0], max_bucket_size=2
    )

    # every item ties (constant similarity) -> side_b would be empty -> guard
    # forces a deterministic alphabetical bisection instead.
    assert buckets == [["a", "b"], ["c", "d"]]


def test_divisive_split_anti_imbalance_guard_triggers_below_threshold_fraction():
    ids = [f"x{i:02d}" for i in range(20)]

    def similarity_fn(a, b):
        if {a, b} == {"x00", "x01"}:
            return 0.0
        return 0.9

    buckets = divisive_split(ids, similarity_fn, lambda group_ids: "x00", max_bucket_size=10)

    # A natural pole-based split would put only "x01" on one side (1/20 = 5%,
    # below the 10% imbalance threshold) -- the guard forces an even bisection.
    assert buckets == [
        [f"x{i:02d}" for i in range(10)],
        [f"x{i:02d}" for i in range(10, 20)],
    ]


def test_divisive_split_is_deterministic_regardless_of_input_order():
    groups = {"a1", "a2", "a3"}, {"b1", "b2", "b3"}
    membership = {item_id: idx for idx, group in enumerate(groups) for item_id in group}

    def similarity_fn(a, b):
        return 1.0 if membership[a] == membership[b] else 0.0

    def pole_a_fn(ids):
        return sorted(ids)[0]

    order_1 = ["a1", "b2", "a3", "b1", "a2", "b3"]
    order_2 = ["b3", "a1", "b1", "a2", "b2", "a3"]

    result_1 = divisive_split(order_1, similarity_fn, pole_a_fn, max_bucket_size=2)
    result_2 = divisive_split(order_2, similarity_fn, pole_a_fn, max_bucket_size=2)

    assert result_1 == result_2


def test_divisive_split_raises_for_invalid_max_bucket_size():
    with pytest.raises(ValueError, match="max_bucket_size"):
        divisive_split(["a"], lambda a, b: 0.0, lambda ids: ids[0], max_bucket_size=0)


# ---------------------------------------------------------------------------
# graph_distance.py
# ---------------------------------------------------------------------------


def test_build_graph_pole_a_fn_picks_highest_entities_weight_tie_broken_by_id():
    entities_by_summary_id = {"s1": {"rare"}, "s2": {"common"}, "s3": {"rare"}}
    idf_weights = {"rare": 2.0, "common": 0.5}

    pole_a_fn = build_graph_pole_a_fn(entities_by_summary_id, idf_weights)

    assert pole_a_fn(["s1", "s2", "s3"]) == "s1"


def test_build_graph_similarity_fn_matches_weighted_jaccard_when_type_and_pattern_weight_zero():
    entities_by_summary_id = {"s1": {"alice", "project-x"}, "s2": {"project-x", "bob"}}
    idf_weights = {"alice": 2.0, "project-x": 1.0, "bob": 3.0}

    similarity_fn = build_graph_similarity_fn(entities_by_summary_id, idf_weights)

    expected = weighted_jaccard({"alice", "project-x"}, {"project-x", "bob"}, idf_weights)
    assert similarity_fn("s1", "s2") == pytest.approx(expected)


def test_build_graph_similarity_fn_blends_type_and_pattern_signals():
    entities_by_summary_id = {"s1": {"alice", "alps"}, "s2": {"bob", "balkans"}}
    idf_weights = {"alice": 1.0, "alps": 1.0, "bob": 1.0, "balkans": 1.0}
    entity_type_by_entity_id = {
        "alice": "person",
        "bob": "person",
        "alps": "location",
        "balkans": "location",
    }
    type_idf_weights = {"person": 1.0, "location": 1.0}
    entity_relations = [("alice", "alps", "goes_to"), ("bob", "balkans", "goes_to")]

    similarity_fn = build_graph_similarity_fn(
        entities_by_summary_id,
        idf_weights,
        entity_type_by_entity_id,
        type_idf_weights,
        entity_weight=0.0,
        type_weight=0.5,
        entity_relations=entity_relations,
        pattern_weight=0.5,
    )

    # zero shared entities (entity_weight is 0 anyway), but identical
    # source/target types and a matching relationship pattern.
    assert similarity_fn("s1", "s2") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# vector_distance.py
# ---------------------------------------------------------------------------


def test_build_vector_pole_a_fn_picks_farthest_item_from_centroid_tie_broken_by_id():
    vectors_by_id = {
        "a": [1.0, 0.0],
        "b": [-0.5, 0.8660254],
        "c": [-0.5, -0.8660254],
    }

    pole_a_fn = build_vector_pole_a_fn(vectors_by_id)

    # the three vectors are symmetric around the origin -> centroid is [0, 0]
    # -> cosine_distance from a zero vector is 1.0 for all three (a genuine
    # tie) -> smallest id wins, regardless of input order.
    assert pole_a_fn(["c", "b", "a"]) == "a"


def test_build_vector_similarity_fn_matches_one_minus_cosine_distance():
    vectors_by_id = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

    similarity_fn = build_vector_similarity_fn(vectors_by_id)

    expected = 1.0 - cosine_distance([1.0, 0.0], [0.0, 1.0])
    assert similarity_fn("a", "b") == pytest.approx(expected)


@pytest.mark.asyncio
async def test_embed_items_for_divisive_split_batches_embed_text_once():
    items = [_summary("s1", text="alpha"), _summary("s2", text="beta")]
    embed_text = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    vector_engine = SimpleNamespace(embedding_engine=SimpleNamespace(embed_text=embed_text))

    vectors_by_id = await embed_items_for_divisive_split(items, vector_engine)

    embed_text.assert_awaited_once_with(["alpha", "beta"])
    assert vectors_by_id == {"s1": [1.0, 0.0], "s2": [0.0, 1.0]}


# ---------------------------------------------------------------------------
# placement.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_divisive_buckets_for_level_graph_signal_routes_misc_summaries_separately():
    entity_items = [_summary(f"s{i}") for i in range(4)]
    misc_items = [_summary(f"m{i}") for i in range(3)]
    entities_by_summary_id = {
        "s0": {"alice"},
        "s1": {"alice"},
        "s2": {"bob"},
        "s3": {"bob"},
        "m0": set(),
        "m1": set(),
        "m2": set(),
    }
    idf_weights = {"alice": 1.0, "bob": 1.0}

    buckets, _ = await build_divisive_buckets_for_level(
        entity_items + misc_items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=2,
        bucketing_strategy="graph",
        vector_engine=None,
        entities_by_summary_id=entities_by_summary_id,
        idf_weights=idf_weights,
    )

    child_sets = {frozenset(bucket.child_ids) for bucket in buckets.values()}
    assert frozenset({"s0", "s1"}) in child_sets
    assert frozenset({"s2", "s3"}) in child_sets
    assert frozenset({"m0", "m1"}) in child_sets
    assert frozenset({"m2"}) in child_sets

    misc_buckets = [b for b in buckets.values() if b.child_ids & {"m0", "m1", "m2"}]
    assert all(bucket.graph_bucket_entity_ids == set() for bucket in misc_buckets)


@pytest.mark.asyncio
async def test_build_divisive_buckets_for_level_respects_max_bucket_size_at_every_leaf():
    items = [_summary(f"s{i}", text=f"s{i}") for i in range(9)]
    vector_engine = _fake_vector_engine({f"s{i}": [float(i), 0.0] for i in range(9)})

    buckets, _ = await build_divisive_buckets_for_level(
        items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=3,
        bucketing_strategy="vector",
        vector_engine=vector_engine,
        entities_by_summary_id={},
        idf_weights={},
    )

    assert all(len(bucket.child_ids) <= 3 for bucket in buckets.values())
    assert sum(len(bucket.child_ids) for bucket in buckets.values()) == 9


@pytest.mark.asyncio
async def test_build_divisive_buckets_for_level_graph_signal_sets_union_entity_ids():
    items = [_summary("s1"), _summary("s2")]
    entities_by_summary_id = {"s1": {"alice"}, "s2": {"alice", "bob"}}
    idf_weights = {"alice": 1.0, "bob": 1.0}

    buckets, _ = await build_divisive_buckets_for_level(
        items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=5,
        bucketing_strategy="graph",
        vector_engine=None,
        entities_by_summary_id=entities_by_summary_id,
        idf_weights=idf_weights,
    )

    assert len(buckets) == 1
    bucket = next(iter(buckets.values()))
    assert bucket.graph_bucket_entity_ids == {"alice", "bob"}


@pytest.mark.asyncio
async def test_build_divisive_buckets_for_level_vector_signal_leaves_entity_ids_none():
    items = [_summary("s1", text="s1"), _summary("s2", text="s2")]
    vector_engine = _fake_vector_engine({"s1": [1.0, 0.0], "s2": [0.0, 1.0]})

    buckets_vector, _ = await build_divisive_buckets_for_level(
        items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=5,
        bucketing_strategy="vector",
        vector_engine=vector_engine,
        entities_by_summary_id={},
        idf_weights={},
    )
    assert all(bucket.graph_bucket_entity_ids is None for bucket in buckets_vector.values())

    # level >= 1 always uses vector, even when bucketing_strategy="graph".
    buckets_level1, _ = await build_divisive_buckets_for_level(
        items,
        level=1,
        dataset_id="dataset-1",
        max_bucket_size=5,
        bucketing_strategy="graph",
        vector_engine=vector_engine,
        entities_by_summary_id={},
        idf_weights={},
    )
    assert all(bucket.graph_bucket_entity_ids is None for bucket in buckets_level1.values())


@pytest.mark.asyncio
async def test_build_divisive_buckets_for_level_bucket_ids_are_deterministic():
    items = [_summary("s1"), _summary("s2")]
    entities_by_summary_id = {"s1": {"alice"}, "s2": {"alice"}}
    idf_weights = {"alice": 1.0}

    buckets, _ = await build_divisive_buckets_for_level(
        items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=5,
        bucketing_strategy="graph",
        vector_engine=None,
        entities_by_summary_id=entities_by_summary_id,
        idf_weights=idf_weights,
    )

    bucket = next(iter(buckets.values()))
    assert bucket.id == str(create_bucket_id("dataset-1", 0, ["s1", "s2"]))


@pytest.mark.asyncio
async def test_build_divisive_buckets_for_level_topic_separability_quality():
    topics = {
        "sports": {"soccer", "goal", "referee"},
        "cooking": {"recipe", "oven", "spice"},
    }
    entities_by_summary_id: dict[str, set[str]] = {}
    true_topic_by_item: dict[str, str] = {}
    items = []
    for topic, entities in topics.items():
        for i in range(6):
            item_id = f"{topic}_{i}"
            items.append(_summary(item_id))
            entities_by_summary_id[item_id] = set(entities)
            true_topic_by_item[item_id] = topic
    idf_weights = {entity: 1.0 for entities in topics.values() for entity in entities}

    buckets, _ = await build_divisive_buckets_for_level(
        items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=3,
        bucketing_strategy="graph",
        vector_engine=None,
        entities_by_summary_id=entities_by_summary_id,
        idf_weights=idf_weights,
    )

    for bucket in buckets.values():
        topics_in_bucket = {true_topic_by_item[child_id] for child_id in bucket.child_ids}
        assert len(topics_in_bucket) == 1


@pytest.mark.asyncio
async def test_build_divisive_buckets_for_level_does_not_mutate_input_summaries():
    items = [_summary("s1"), _summary("s2")]
    entities_by_summary_id = {"s1": {"alice"}, "s2": {"alice"}}
    idf_weights = {"alice": 1.0}
    original_ids = [item.id for item in items]
    original_texts = [item.text for item in items]

    await build_divisive_buckets_for_level(
        items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=5,
        bucketing_strategy="graph",
        vector_engine=None,
        entities_by_summary_id=entities_by_summary_id,
        idf_weights=idf_weights,
    )

    assert [item.id for item in items] == original_ids
    assert [item.text for item in items] == original_texts
    assert all(item.global_context_bucket_id is None for item in items)


# ---------------------------------------------------------------------------
# Acceptance criterion: divisively-built buckets extend via the existing,
# unmodified incremental placement code with zero id churn.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_divisively_built_graph_bucket_extends_via_place_graph_summaries_incrementally_with_zero_id_churn():
    items = [_summary("s1"), _summary("s2")]
    entities_by_summary_id = {"s1": {"alice"}, "s2": {"alice"}, "s3": {"alice"}}
    idf_weights = {"alice": 1.0}

    buckets, _ = await build_divisive_buckets_for_level(
        items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=5,
        bucketing_strategy="graph",
        vector_engine=None,
        entities_by_summary_id=entities_by_summary_id,
        idf_weights=idf_weights,
    )
    existing_bucket = next(iter(buckets.values()))
    original_bucket_id = existing_bucket.id

    new_summary = _summary("s3")
    place_graph_summaries_incrementally(
        [new_summary],
        [existing_bucket],
        entities_by_summary_id,
        idf_weights,
        dataset_id="dataset-1",
        level=0,
        max_bucket_size=5,
        min_overlap=0.05,
    )

    assert existing_bucket.id == original_bucket_id
    assert existing_bucket.child_ids == {"s1", "s2", "s3"}


@pytest.mark.asyncio
async def test_divisively_built_vector_bucket_extends_via_assign_items_to_buckets_with_zero_id_churn():
    items = [_summary("s1", text="s1"), _summary("s2", text="s2")]
    build_vector_engine = _fake_vector_engine({"s1": [1.0, 0.0], "s2": [1.0, 0.0]})

    buckets, _ = await build_divisive_buckets_for_level(
        items,
        level=0,
        dataset_id="dataset-1",
        max_bucket_size=5,
        bucketing_strategy="vector",
        vector_engine=build_vector_engine,
        entities_by_summary_id={},
        idf_weights={},
    )
    existing_bucket = next(iter(buckets.values()))
    original_bucket_id = existing_bucket.id
    assert existing_bucket.child_ids == {"s1", "s2"}

    class _NearestResult:
        def __init__(self, id_: str, score: float):
            self.id = id_
            self.score = score

    search_results = [_NearestResult("s1", 0.1), _NearestResult("s2", 0.1)]
    search_vector_engine = SimpleNamespace(search=AsyncMock(return_value=search_results))
    new_summary = _summary("s3", text="s3")

    await assign_items_to_buckets(
        [new_summary],
        [existing_bucket],
        level=0,
        dataset_id="dataset-1",
        vector_engine=search_vector_engine,
        source_collection="TextSummary_text",
        max_bucket_size=5,
        placement_distance_threshold=0.5,
    )

    assert existing_bucket.id == original_bucket_id
    assert existing_bucket.child_ids == {"s1", "s2", "s3"}


# ---------------------------------------------------------------------------
# Dispatch-level tests
# ---------------------------------------------------------------------------


def _build_options(**overrides) -> BuildOptions:
    defaults = dict(
        dataset_id="dataset-1",
        vector_engine=None,
        max_bucket_size=5,
        placement_distance_threshold=0.5,
        bucketing_strategy="vector",
        min_overlap=0.05,
        entities_by_summary_id={},
        idf_weights={},
        entity_type_by_entity_id={},
        type_idf_weights={},
        entity_weight=1.0,
        type_weight=0.0,
        pattern_weight=0.0,
        entity_relations=[],
        edge_type_embeddings={},
        pattern_distance_threshold=0.5,
        build_strategy="seed_and_absorb",
        is_first_build=False,
        ctx=None,
    )
    defaults.update(overrides)
    return BuildOptions(**defaults)


@pytest.mark.asyncio
async def test_place_items_for_level_dispatches_to_divisive_when_requested_and_no_existing_buckets(
    monkeypatch,
):
    calls = []

    async def fake_divisive(all_items, level, options):
        calls.append("divisive")
        return {}, []

    monkeypatch.setattr(build_module, "place_divisive_items", fake_divisive)

    options = _build_options(build_strategy="divisive", is_first_build=True)
    await place_items_for_level([], [], [], level=0, options=options)

    assert calls == ["divisive"]


@pytest.mark.asyncio
async def test_place_items_for_level_ignores_divisive_when_existing_buckets_present(monkeypatch):
    calls = []

    async def fake_divisive(all_items, level, options):
        calls.append("divisive")
        return {}, []

    def fake_graph_items(changed_items, all_items, existing_buckets, level, options):
        calls.append("graph")
        return {}, []

    monkeypatch.setattr(build_module, "place_divisive_items", fake_divisive)
    monkeypatch.setattr(build_module, "place_graph_items", fake_graph_items)

    options = _build_options(
        bucketing_strategy="graph", build_strategy="divisive", is_first_build=True
    )
    existing_bucket = _bucket("bucket-1", {"s1"}, {"alice"})

    await place_items_for_level([], [], [existing_bucket], level=0, options=options)

    assert calls == ["graph"]


def test_validate_global_context_index_config_rejects_invalid_build_strategy():
    with pytest.raises(ValueError, match="build_strategy"):
        validate_global_context_index_config(20, 0.5, "vector", 0.05, build_strategy="invalid")


@pytest.mark.asyncio
async def test_build_and_persist_context_index_threads_build_strategy(monkeypatch):
    captured = {}

    async def fake_build_context_index(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(update_module, "build_context_index", fake_build_context_index)
    monkeypatch.setattr(update_module, "persist_context_index_edges", AsyncMock())

    scope = SimpleNamespace(
        text_summaries=[],
        dataset_id="dataset-1",
        context_input=SimpleNamespace(buckets=[], root=None),
    )
    unified_engine = SimpleNamespace(vector=SimpleNamespace())

    await build_and_persist_context_index(
        scope, [], unified_engine, 20, 0.5, "graph", 0.05, None, None, build_strategy="divisive"
    )

    assert captured["build_strategy"] == "divisive"

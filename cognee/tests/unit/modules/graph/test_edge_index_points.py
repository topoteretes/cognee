import pytest

from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.graph.utils.edge_index_points import (
    build_edge_index_points,
    edge_instance_id,
)


def test_distinct_edge_texts_share_one_type_but_keep_two_instances():
    edges = [
        (
            "source-a",
            "target-a",
            "depends_on",
            {
                "edge_object_id": "15bfc0f0-51d7-5ac8-8589-3c32fe75aa10",
                "edge_text": "Package A depends on Package B for builds.",
            },
        ),
        (
            "source-c",
            "target-c",
            "depends_on",
            {
                "edge_object_id": "f0388896-5f24-5d7f-a919-b5d49f6af57e",
                "edge_text": "Service C depends on Service D at runtime.",
            },
        ),
    ]

    points = build_edge_index_points(edges)

    assert [(point.relationship_name, point.number_of_edges) for point in points.edge_types] == [
        ("depends_on", 2)
    ]
    assert [point.text for point in points.edge_instances] == [
        "Package A depends on Package B for builds.",
        "Service C depends on Service D at runtime.",
    ]


def test_edge_text_change_preserves_structural_instance_id():
    first = build_edge_index_points(
        [("source", "target", "depends_on", {"edge_text": "Old explanation."})]
    )
    second = build_edge_index_points(
        [("source", "target", "depends_on", {"edge_text": "New explanation."})]
    )

    assert first.edge_instances[0].id == second.edge_instances[0].id
    assert first.edge_instances[0].text == "Old explanation."
    assert second.edge_instances[0].text == "New explanation."


def test_blank_edge_text_falls_back_without_changing_id():
    points = build_edge_index_points([("source", "target", "depends_on", {"edge_text": "   "})])

    assert points.edge_instances[0].text == "depends_on"
    assert str(points.edge_instances[0].id) == generate_edge_object_id(
        "source", "target", "depends_on"
    )


def test_reverse_direction_and_changed_relationship_have_distinct_instance_ids():
    points = build_edge_index_points(
        [
            ("source", "target", "depends_on", {}),
            ("target", "source", "depends_on", {}),
            ("source", "target", "contains", {}),
        ]
    )

    assert [str(point.id) for point in points.edge_instances] == [
        "7c3827c3-8990-54b2-b9d9-0cfac2c54955",
        "1217d0f8-8088-5222-9e11-f497f99db756",
        "d45707e5-9d5a-547d-aba9-839b00079ef0",
    ]


def test_duplicate_structural_edge_uses_last_text_but_keeps_local_type_count():
    points = build_edge_index_points(
        [
            ("source", "target", "depends_on", {"edge_text": "First text."}),
            ("source", "target", "depends_on", {"edge_text": "Last text."}),
        ]
    )

    assert [(point.relationship_name, point.number_of_edges) for point in points.edge_types] == [
        ("depends_on", 2)
    ]
    assert [point.text for point in points.edge_instances] == ["Last text."]


@pytest.mark.parametrize("relationship_name", [None, "", "   "])
def test_blank_relationship_name_is_rejected(relationship_name):
    with pytest.raises(ValueError, match="relationship_name"):
        build_edge_index_points([("source", "target", relationship_name, {})])


def test_relationship_counts_override_local_batch_count_and_names_are_trimmed():
    points = build_edge_index_points(
        [("source", "target", " depends_on ", {"edge_text": "A depends on B."})],
        relationship_counts={"depends_on": 9},
    )

    assert [(point.relationship_name, point.number_of_edges) for point in points.edge_types] == [
        ("depends_on", 9)
    ]
    assert points.edge_instances[0].relationship_name == "depends_on"
    assert str(points.edge_instances[0].id) == generate_edge_object_id(
        "source", "target", "depends_on"
    )


def test_edge_instance_id_prefers_stored_edge_object_id():
    assert (
        edge_instance_id(
            "source",
            "target",
            "depends_on",
            {"edge_object_id": "15bfc0f0-51d7-5ac8-8589-3c32fe75aa10"},
        )
        == "15bfc0f0-51d7-5ac8-8589-3c32fe75aa10"
    )

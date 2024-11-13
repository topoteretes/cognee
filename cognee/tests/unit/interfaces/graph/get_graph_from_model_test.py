from cognee.tests.unit.interfaces.graph.util import run_test_against_ground_truth

EDGE_GROUND_TRUTH = (
    "boris",
    "car1",
    "owns_car",
    {
        "source_node_id": "boris",
        "target_node_id": "car1",
        "relationship_name": "owns_car",
        "metadata": {"type": "list"},
    },
)

CAR_GROUND_TRUTH = {
    "id": "car1",
    "brand": "Toyota",
    "model": "Camry",
    "year": 2020,
    "color": "Blue",
}

PERSON_GROUND_TRUTH = {
    "id": "boris",
    "name": "Boris",
    "age": 30,
    "driving_license": {
        "issued_by": "PU Vrsac",
        "issued_on": "2025-11-06",
        "number": "1234567890",
        "expires_on": "2025-11-06",
    },
}


def test_extracted_person(graph_outputs):
    (_, person, _, _) = graph_outputs

    run_test_against_ground_truth("person", person, PERSON_GROUND_TRUTH)


def test_extracted_car(graph_outputs):
    (car, _, _, _) = graph_outputs
    run_test_against_ground_truth("car", car, CAR_GROUND_TRUTH)


def test_extracted_edge(graph_outputs):
    (_, _, edge, _) = graph_outputs

    assert (
        EDGE_GROUND_TRUTH[:3] == edge[:3]
    ), f"{EDGE_GROUND_TRUTH[:3] = } != {edge[:3] = }"
    for key, ground_truth in EDGE_GROUND_TRUTH[3].items():
        assert ground_truth == edge[3][key], f"{ground_truth = } != {edge[3][key] = }"

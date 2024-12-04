from cognee.modules.graph.utils import get_graph_from_model
from cognee.tests.unit.interfaces.graph.util import run_test_against_ground_truth

CAR_SEDAN_EDGE = (
    "car1",
    "sedan",
    "is_type",
    {
        "source_node_id": "car1",
        "target_node_id": "sedan",
        "relationship_name": "is_type",
    },
)


BORIS_CAR_EDGE_GROUND_TRUTH = (
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

CAR_TYPE_GROUND_TRUTH = {"id": "sedan"}

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


def test_extracted_car_type(boris):
    nodes, _ = get_graph_from_model(boris)
    assert len(nodes) == 3
    car_type = nodes[0]
    run_test_against_ground_truth("car_type", car_type, CAR_TYPE_GROUND_TRUTH)


def test_extracted_car(boris):
    nodes, _ = get_graph_from_model(boris)
    assert len(nodes) == 3
    car = nodes[1]
    run_test_against_ground_truth("car", car, CAR_GROUND_TRUTH)


def test_extracted_person(boris):
    nodes, _ = get_graph_from_model(boris)
    assert len(nodes) == 3
    person = nodes[2]
    run_test_against_ground_truth("person", person, PERSON_GROUND_TRUTH)


def test_extracted_car_sedan_edge(boris):
    _, edges = get_graph_from_model(boris)
    edge = edges[0]

    assert CAR_SEDAN_EDGE[:3] == edge[:3], f"{CAR_SEDAN_EDGE[:3] = } != {edge[:3] = }"
    for key, ground_truth in CAR_SEDAN_EDGE[3].items():
        assert ground_truth == edge[3][key], f"{ground_truth = } != {edge[3][key] = }"


def test_extracted_boris_car_edge(boris):
    _, edges = get_graph_from_model(boris)
    edge = edges[1]

    assert (
        BORIS_CAR_EDGE_GROUND_TRUTH[:3] == edge[:3]
    ), f"{BORIS_CAR_EDGE_GROUND_TRUTH[:3] = } != {edge[:3] = }"
    for key, ground_truth in BORIS_CAR_EDGE_GROUND_TRUTH[3].items():
        assert ground_truth == edge[3][key], f"{ground_truth = } != {edge[3][key] = }"

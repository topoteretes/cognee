from cognee.modules.graph.utils import (
    get_graph_from_model,
    get_model_instance_from_graph,
)
from cognee.tests.unit.interfaces.graph.util import run_test_against_ground_truth

PARSED_PERSON_GROUND_TRUTH = {
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

CAR_GROUND_TRUTH = {
    "id": "car1",
    "brand": "Toyota",
    "model": "Camry",
    "year": 2020,
    "color": "Blue",
}


def test_parsed_person(boris):
    nodes, edges = get_graph_from_model(boris)
    parsed_person = get_model_instance_from_graph(nodes, edges, "boris")

    run_test_against_ground_truth(
        "parsed_person", parsed_person, PARSED_PERSON_GROUND_TRUTH
    )
    run_test_against_ground_truth("car", parsed_person.owns_car[0], CAR_GROUND_TRUTH)

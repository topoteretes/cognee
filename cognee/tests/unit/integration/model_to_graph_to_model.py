from enum import Enum
from datetime import datetime, timezone
from typing import Optional
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import get_graph_from_model, get_model_instance_from_graph


EDGE_GROUND_TRUTH = (
    "boris",
    "car1",
    "owns_car",
    {'source_node_id': 'boris', 'target_node_id': 'car1', 'relationship_name': 'owns_car', 'metadata': {'type': 'list'}}
)

CAR_GROUND_TRUTH = {
    "id": "car1",
    "brand": "Toyota",
    "model": "Camry",
    "year": 2020,
    "color": "Blue"
}

PERSON_GROUND_TRUTH = {
    "id": "boris",
    "name": "Boris",
    "age": 30,
    "driving_license": {'issued_by': "PU Vrsac", 'issued_on': '2025-11-06', 'number': '1234567890', 'expires_on': '2025-11-06'}
}


PARSED_PERSON_GROUND_TRUTH = {
    "id": "boris",
    "name": "Boris",
    "age": 30,
    "driving_license": {'issued_by': 'PU Vrsac', 'issued_on': '2025-11-06', 'number': '1234567890', 'expires_on': '2025-11-06'},
}


class CarTypeName(Enum):
    Pickup = "Pickup"
    Sedan = "Sedan"
    SUV = "SUV"
    Coupe = "Coupe"
    Convertible = "Convertible"
    Hatchback = "Hatchback"
    Wagon = "Wagon"
    Minivan = "Minivan"
    Van = "Van"

class CarType(DataPoint):
    id: str
    name: CarTypeName
    _metadata: dict = dict(index_fields = ["name"])

class Car(DataPoint):
    id: str
    brand: str
    model: str
    year: int
    color: str
    is_type: CarType

class Person(DataPoint):
    id: str
    name: str
    age: int
    owns_car: list[Car]
    driving_license: Optional[dict]
    _metadata: dict = dict(index_fields = ["name"])




if __name__ == "__main__":

    boris = Person(
        id = "boris",
        name = "Boris",
        age = 30,
        owns_car = [Car(
            id = "car1",
            brand = "Toyota",
            model = "Camry",
            year = 2020,
            color = "Blue",
            is_type = CarType(id = "sedan", name = CarTypeName.Sedan),
        )],
        driving_license = {
            "issued_by": "PU Vrsac",
            "issued_on": "2025-11-06",
            "number": "1234567890",
            "expires_on": "2025-11-06",
        },
    )

    nodes, edges = get_graph_from_model(boris)

    car, person = nodes[0], nodes[1]
    edge = edges[0]

    def test_against_ground_truth(test_target_item_name, test_target_item, ground_truth_dict):
        for key, ground_truth in ground_truth_dict.items():
            if isinstance(ground_truth, dict):
                for key2, ground_truth2 in ground_truth.items():
                    assert ground_truth2 == getattr(test_target_item, key)[key2], f'{test_target_item_name}/{key = }/{key2 = }: {ground_truth2 = } != {getattr(test_target_item, key)[key2] = }'
            else:
                assert ground_truth == getattr(test_target_item, key), f'{test_target_item_name}/{key = }: {ground_truth = } != {getattr(test_target_item, key) = }'
        time_delta = datetime.now(timezone.utc) - getattr(test_target_item, "updated_at")

        assert time_delta.total_seconds() < 20, f"{ time_delta.total_seconds() = }"

    test_against_ground_truth("car", car, CAR_GROUND_TRUTH)
    test_against_ground_truth("person", person, PERSON_GROUND_TRUTH)

    assert EDGE_GROUND_TRUTH[:3] == edge[:3], f'{EDGE_GROUND_TRUTH[:3] = } != {edge[:3] = }'
    for key, ground_truth in EDGE_GROUND_TRUTH[3].items():
        assert ground_truth == edge[3][key], f'{ground_truth = } != {edge[3][key] = }'

    parsed_person = get_model_instance_from_graph(nodes, edges, 'boris')

    test_against_ground_truth("parsed_person", parsed_person, PARSED_PERSON_GROUND_TRUTH)
    test_against_ground_truth("car", parsed_person.owns_car[0], CAR_GROUND_TRUTH)

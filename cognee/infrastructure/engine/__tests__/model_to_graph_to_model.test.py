from enum import Enum
from typing import Optional
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import get_graph_from_model, get_model_instance_from_graph


if __name__ == "__main__":

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
        driving_licence: Optional[dict]
        _metadata: dict = dict(index_fields = ["name"])

    boris = Person(
        id = "boris",
        name = "Boris",
        age = 30,
        owns_car = [
            Car(
                id = "car1",
                brand = "Toyota",
                model = "Camry",
                year = 2020,
                color = "Blue",
                is_type = CarType(id = "sedan", name = CarTypeName.Sedan),
            ),
        ],
        driving_licence = {
            "issued_by": "PU Vrsac",
            "issued_on": "2025-11-06",
            "number": "1234567890",
            "expires_on": "2025-11-06",
        },
    )

    nodes, edges = get_graph_from_model(boris)

    print(nodes)
    print(edges)

    person_data = nodes[len(nodes) - 1]

    parsed_person = get_model_instance_from_graph(nodes, edges, 'boris')

    print(parsed_person)
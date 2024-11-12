from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import (
    get_graph_from_model,
    get_model_instance_from_graph,
)


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
    _metadata: dict = dict(index_fields=["name"])


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
    _metadata: dict = dict(index_fields=["name"])


@pytest.fixture(scope="session")
def graph_outputs():
    boris = Person(
        id="boris",
        name="Boris",
        age=30,
        owns_car=[
            Car(
                id="car1",
                brand="Toyota",
                model="Camry",
                year=2020,
                color="Blue",
                is_type=CarType(id="sedan", name=CarTypeName.Sedan),
            )
        ],
        driving_license={
            "issued_by": "PU Vrsac",
            "issued_on": "2025-11-06",
            "number": "1234567890",
            "expires_on": "2025-11-06",
        },
    )
    nodes, edges = get_graph_from_model(boris)

    car, person = nodes[0], nodes[1]
    edge = edges[0]

    parsed_person = get_model_instance_from_graph(nodes, edges, "boris")

    return (car, person, edge, parsed_person)

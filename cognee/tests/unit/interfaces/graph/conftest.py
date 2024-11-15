from enum import Enum
from typing import Optional

import pytest

from cognee.infrastructure.engine import DataPoint


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


@pytest.fixture(scope="function")
def boris():
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
    return boris

import random
import string
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
    age: Optional[int]
    owns_car: Optional[list[Car]]
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


class Organization(DataPoint):
    id: str
    name: str
    members: Optional[list["SocietyPerson"]]


class SocietyPerson(DataPoint):
    id: str
    name: str
    memberships: Optional[list[Organization]]


Organization.model_rebuild()
SocietyPerson.model_rebuild()


ORGANIZATION_NAMES = [
    "ChessClub",
    "RowingClub",
    "TheatreTroupe",
    "PoliticalParty",
    "Charity",
    "FanClub",
    "FilmClub",
    "NeighborhoodGroup",
    "LocalCouncil",
    "Band",
]
PERSON_NAMES = ["Sarah", "Anna", "John", "Sam"]


def create_society_person_recursive(id, name, organization_names, max_depth, depth=0):
    if depth < max_depth:
        memberships = [
            create_organization_recursive(
                org_name, org_name.lower(), PERSON_NAMES, max_depth, depth + 1
            )
            for org_name in organization_names
        ]
    else:
        memberships = None

    id_suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    return SocietyPerson(
        id=f"{id}{depth}-{id_suffix}", name=f"{name}{depth}", memberships=memberships
    )


def create_organization_recursive(id, name, member_names, max_depth, depth=0):
    if depth < max_depth:
        members = [
            create_society_person_recursive(
                member_name,
                member_name.lower(),
                ORGANIZATION_NAMES,
                max_depth,
                depth + 1,
            )
            for member_name in member_names
        ]
    else:
        members = None

    id_suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    return Organization(
        id=f"{id}{depth}-{id_suffix}", name=f"{id}{name}", members=members
    )


def count_society(obj):
    if isinstance(obj, SocietyPerson):
        if obj.memberships is not None:
            organization_counts, society_person_counts = zip(
                *[count_society(organization) for organization in obj.memberships]
            )
            organization_count = sum(organization_counts)
            society_person_count = sum(society_person_counts) + 1
            return (organization_count, society_person_count)
        else:
            return (0, 1)
    if isinstance(obj, Organization):
        if obj.members is not None:
            organization_counts, society_person_counts = zip(
                *[count_society(organization) for organization in obj.members]
            )
            organization_count = sum(organization_counts) + 1
            society_person_count = sum(society_person_counts)
            return (organization_count, society_person_count)
        else:
            return (1, 0)
    else:
        return (0, 0)


@pytest.fixture(scope="function")
def society():
    society = create_organization_recursive("society", "Society", PERSON_NAMES, 4)

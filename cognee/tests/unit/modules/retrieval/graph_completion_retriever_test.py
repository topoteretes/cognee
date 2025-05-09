import os
import pytest
import pathlib
from typing import Optional, Union

import cognee
from cognee.low_level import setup, DataPoint
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


class TestGraphCompletionRetriever:
    @pytest.mark.asyncio
    async def test_graph_completion_context_simple(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_graph_context"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_graph_context"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        class Company(DataPoint):
            name: str

        class Person(DataPoint):
            name: str
            works_for: Company

        company1 = Company(name="Figma")
        company2 = Company(name="Canva")
        person1 = Person(name="Steve Rodger", works_for=company1)
        person2 = Person(name="Ike Loma", works_for=company1)
        person3 = Person(name="Jason Statham", works_for=company1)
        person4 = Person(name="Mike Broski", works_for=company2)
        person5 = Person(name="Christina Mayer", works_for=company2)

        entities = [company1, company2, person1, person2, person3, person4, person5]

        await add_data_points(entities)

        retriever = GraphCompletionRetriever()

        context = await retriever.get_context("Who works at Canva?")

        assert "Mike Broski --[works_for]--> Canva" in context, "Failed to get Mike Broski"
        assert "Christina Mayer --[works_for]--> Canva" in context, "Failed to get Christina Mayer"

    @pytest.mark.asyncio
    async def test_graph_completion_context_complex(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_graph_completion_context"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_graph_completion_context"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        class Company(DataPoint):
            name: str
            metadata: dict = {"index_fields": ["name"]}

        class Car(DataPoint):
            brand: str
            model: str
            year: int

        class Location(DataPoint):
            country: str
            city: str

        class Home(DataPoint):
            location: Location
            rooms: int
            sqm: int

        class Person(DataPoint):
            name: str
            works_for: Company
            owns: Optional[list[Union[Car, Home]]] = None

        company1 = Company(name="Figma")
        company2 = Company(name="Canva")

        person1 = Person(name="Mike Rodger", works_for=company1)
        person1.owns = [Car(brand="Toyota", model="Camry", year=2020)]

        person2 = Person(name="Ike Loma", works_for=company1)
        person2.owns = [
            Car(brand="Tesla", model="Model S", year=2021),
            Home(location=Location(country="USA", city="New York"), sqm=80, rooms=4),
        ]

        person3 = Person(name="Jason Statham", works_for=company1)

        person4 = Person(name="Mike Broski", works_for=company2)
        person4.owns = [Car(brand="Ford", model="Mustang", year=1978)]

        person5 = Person(name="Christina Mayer", works_for=company2)
        person5.owns = [Car(brand="Honda", model="Civic", year=2023)]

        entities = [company1, company2, person1, person2, person3, person4, person5]

        await add_data_points(entities)

        retriever = GraphCompletionRetriever(top_k=20)

        context = await retriever.get_context("Who works at Figma?")

        print(context)

        assert "Mike Rodger --[works_for]--> Figma" in context, "Failed to get Mike Rodger"
        assert "Ike Loma --[works_for]--> Figma" in context, "Failed to get Ike Loma"
        assert "Jason Statham --[works_for]--> Figma" in context, "Failed to get Jason Statham"

    @pytest.mark.asyncio
    async def test_get_graph_completion_context_on_empty_graph(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_graph_completion_context"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_graph_completion_context"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        retriever = GraphCompletionRetriever()

        with pytest.raises(DatabaseNotCreatedError):
            await retriever.get_context("Who works at Figma?")

        await setup()

        context = await retriever.get_context("Who works at Figma?")
        assert context == "", "Context should be empty on an empty graph"


if __name__ == "__main__":
    from asyncio import run

    test = TestGraphCompletionRetriever()

    async def main():
        await test.test_graph_completion_context_simple()
        await test.test_graph_completion_context_complex()
        await test.test_get_graph_completion_context_on_empty_graph()

    run(main())

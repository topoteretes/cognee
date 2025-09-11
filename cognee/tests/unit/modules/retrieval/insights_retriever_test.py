import os
import pytest
import pathlib

import cognee
from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.modules.engine.models import Entity, EntityType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.insights_retriever import InsightsRetriever


class TestInsightsRetriever:
    @pytest.mark.asyncio
    async def test_insights_context_simple(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_insights_context_simple"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_insights_context_simple"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        entityTypePerson = EntityType(
            name="Person",
            description="An individual",
        )

        person1 = Entity(
            name="Steve Rodger",
            is_a=entityTypePerson,
            description="An American actor, comedian, and filmmaker",
        )

        person2 = Entity(
            name="Mike Broski",
            is_a=entityTypePerson,
            description="Financial advisor and philanthropist",
        )

        person3 = Entity(
            name="Christina Mayer",
            is_a=entityTypePerson,
            description="Maker of next generation of iconic American music videos",
        )

        entityTypeCompany = EntityType(
            name="Company",
            description="An organization that operates on an annual basis",
        )

        company1 = Entity(
            name="Apple",
            is_a=entityTypeCompany,
            description="An American multinational technology company headquartered in Cupertino, California",
        )

        company2 = Entity(
            name="Google",
            is_a=entityTypeCompany,
            description="An American multinational technology company that specializes in Internet-related services and products",
        )

        company3 = Entity(
            name="Facebook",
            is_a=entityTypeCompany,
            description="An American social media, messaging, and online platform",
        )

        entities = [person1, person2, person3, company1, company2, company3]

        await add_data_points(entities)

        retriever = InsightsRetriever()

        context = await retriever.get_context("Mike")

        assert context[0][0]["name"] == "Mike Broski", "Failed to get Mike Broski"

    @pytest.mark.asyncio
    async def test_insights_context_complex(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_insights_context_complex"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_insights_context_complex"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        entityTypePerson = EntityType(
            name="Person",
            description="An individual",
        )

        person1 = Entity(
            name="Steve Rodger",
            is_a=entityTypePerson,
            description="An American actor, comedian, and filmmaker",
        )

        person2 = Entity(
            name="Mike Broski",
            is_a=entityTypePerson,
            description="Financial advisor and philanthropist",
        )

        person3 = Entity(
            name="Christina Mayer",
            is_a=entityTypePerson,
            description="Maker of next generation of iconic American music videos",
        )

        person4 = Entity(
            name="Jason Statham",
            is_a=entityTypePerson,
            description="An American actor",
        )

        person5 = Entity(
            name="Mike Tyson",
            is_a=entityTypePerson,
            description="A former professional boxer from the United States",
        )

        entityTypeCompany = EntityType(
            name="Company",
            description="An organization that operates on an annual basis",
        )

        company1 = Entity(
            name="Apple",
            is_a=entityTypeCompany,
            description="An American multinational technology company headquartered in Cupertino, California",
        )

        company2 = Entity(
            name="Google",
            is_a=entityTypeCompany,
            description="An American multinational technology company that specializes in Internet-related services and products",
        )

        company3 = Entity(
            name="Facebook",
            is_a=entityTypeCompany,
            description="An American social media, messaging, and online platform",
        )

        entities = [person1, person2, person3, company1, company2, company3]

        await add_data_points(entities)

        graph_engine = await get_graph_engine()

        await graph_engine.add_edges(
            [
                (
                    (str)(person1.id),
                    (str)(company1.id),
                    "works_for",
                    dict(
                        relationship_name="works_for",
                        source_node_id=person1.id,
                        target_node_id=company1.id,
                    ),
                ),
                (
                    (str)(person2.id),
                    (str)(company2.id),
                    "works_for",
                    dict(
                        relationship_name="works_for",
                        source_node_id=person2.id,
                        target_node_id=company2.id,
                    ),
                ),
                (
                    (str)(person3.id),
                    (str)(company3.id),
                    "works_for",
                    dict(
                        relationship_name="works_for",
                        source_node_id=person3.id,
                        target_node_id=company3.id,
                    ),
                ),
                (
                    (str)(person4.id),
                    (str)(company1.id),
                    "works_for",
                    dict(
                        relationship_name="works_for",
                        source_node_id=person4.id,
                        target_node_id=company1.id,
                    ),
                ),
                (
                    (str)(person5.id),
                    (str)(company1.id),
                    "works_for",
                    dict(
                        relationship_name="works_for",
                        source_node_id=person5.id,
                        target_node_id=company1.id,
                    ),
                ),
            ]
        )

        retriever = InsightsRetriever(top_k=20)

        context = await retriever.get_context("Christina")

        assert context[0][0]["name"] == "Christina Mayer", "Failed to get Christina Mayer"

    @pytest.mark.asyncio
    async def test_insights_context_on_empty_graph(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_insights_context_on_empty_graph"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_insights_context_on_empty_graph"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        retriever = InsightsRetriever()

        with pytest.raises(NoDataError):
            await retriever.get_context("Christina Mayer")

        vector_engine = get_vector_engine()
        await vector_engine.create_collection("Entity_name", payload_schema=Entity)
        await vector_engine.create_collection("EntityType_name", payload_schema=EntityType)

        context = await retriever.get_context("Christina Mayer")
        assert context == [], "Returned context should be empty on an empty graph"

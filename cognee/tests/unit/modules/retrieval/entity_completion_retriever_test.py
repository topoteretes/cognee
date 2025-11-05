import os
import pytest
import pathlib
from pydantic import BaseModel

import cognee
from cognee.low_level import setup
from cognee.tasks.storage import add_data_points
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.retrieval.EntityCompletionRetriever import EntityCompletionRetriever
from cognee.modules.retrieval.entity_extractors.DummyEntityExtractor import DummyEntityExtractor
from cognee.modules.retrieval.context_providers.DummyContextProvider import DummyContextProvider


class TestAnswer(BaseModel):
    answer: str
    explanation: str


# TODO: Add more tests, similar to other retrievers.
# TODO: For the tests, one needs to define an Entity Extractor and a Context Provider.
class TestEntityCompletionRetriever:
    @pytest.mark.asyncio
    async def test_get_entity_structured_completion(self):
        system_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".cognee_system/test_get_entity_structured_completion"
        )
        cognee.config.system_root_directory(system_directory_path)
        data_directory_path = os.path.join(
            pathlib.Path(__file__).parent, ".data_storage/test_get_entity_structured_completion"
        )
        cognee.config.data_root_directory(data_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await setup()

        entity_type = EntityType(name="Person", description="A human individual")
        entity = Entity(name="Albert Einstein", is_a=entity_type, description="A famous physicist")

        entities = [entity]
        await add_data_points(entities)

        retriever = EntityCompletionRetriever(DummyEntityExtractor(), DummyContextProvider())

        # Test with string response model (default)
        string_answer = await retriever.get_completion("Who is Albert Einstein?")
        assert isinstance(string_answer, list), f"Expected str, got {type(string_answer).__name__}"
        assert all(isinstance(item, str) and item.strip() for item in string_answer), (
            "Answer should not be empty"
        )

        # Test with structured response model
        structured_answer = await retriever.get_completion(
            "Who is Albert Einstein?", response_model=TestAnswer
        )
        assert isinstance(structured_answer, list), (
            f"Expected list, got {type(structured_answer).__name__}"
        )
        assert all(isinstance(item, TestAnswer) for item in structured_answer), (
            f"Expected TestAnswer, got {type(structured_answer).__name__}"
        )

        assert structured_answer[0].answer.strip(), "Answer field should not be empty"
        assert structured_answer[0].explanation.strip(), "Explanation field should not be empty"

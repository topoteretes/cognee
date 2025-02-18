from typing import List, Type
import logging

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.tasks.entity_completion.context_providers.dummy_context_provider import (
    DummyContextProvider,
)
from cognee.tasks.entity_completion.entity_completion_config import EntityCompletionConfig
from cognee.tasks.entity_completion.entity_extractors.base_entity_extractor import (
    BaseEntityExtractor,
)
from cognee.tasks.entity_completion.context_providers.base_context_provider import (
    BaseContextProvider,
)
from cognee.tasks.entity_completion.entity_extractors.dummy_entity_extractor import (
    DummyEntityExtractor,
)

logger = logging.getLogger(__name__)

entity_completion_config = EntityCompletionConfig()


async def get_llm_response(query: str, context: str) -> str:
    """Generate LLM response based on query and context."""
    try:
        args = {
            "question": query,
            "context": context,
        }
        user_prompt = render_prompt(entity_completion_config.user_prompt_template, args)
        system_prompt = read_query_prompt(entity_completion_config.system_prompt_template)

        llm_client = get_llm_client()
        return await llm_client.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=str,
        )
    except Exception as e:
        logger.error(f"LLM response generation failed: {str(e)}")
        raise


async def entity_completion(
    query: str, extractor: Type[BaseEntityExtractor], getter: Type[BaseContextProvider]
) -> List[str]:
    """Execute entity-based completion using configurable components."""
    if not query or not isinstance(query, str):
        logger.error("Invalid query type or empty query")
        return ["Invalid query input"]

    try:
        logger.info(f"Processing query: {query[:100]}")
        entities = await extractor().extract_entities(query)
        logger.debug(f"Extracted entities: {[e.name for e in entities]}")

        if not entities:
            logger.info("No entities extracted")
            return ["No entities found"]

        context = await getter().get_context(entities, query)

        if not context:
            logger.info("No context retrieved")
            return ["No context found"]

        return [await get_llm_response(query, context)]

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return ["Pipeline execution failed"]


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def run_entity_completion():
        result = await entity_completion(
            "Tell me about Einstein", extractor=DummyEntityExtractor, getter=DummyContextProvider
        )
        print(f"Query Response: {result[0]}")

    asyncio.run(run_entity_completion())

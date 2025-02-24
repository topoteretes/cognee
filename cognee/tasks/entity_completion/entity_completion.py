from typing import List
import logging

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.infrastructure.entities.BaseEntityExtractor import (
    BaseEntityExtractor,
)
from cognee.infrastructure.context.BaseContextProvider import (
    BaseContextProvider,
)

logger = logging.getLogger("entity_completion")

# Default prompt template paths
DEFAULT_SYSTEM_PROMPT_TEMPLATE = "answer_simple_question.txt"
DEFAULT_USER_PROMPT_TEMPLATE = "context_for_question.txt"


async def get_llm_response(
    query: str,
    context: str,
    system_prompt_template: str = None,
    user_prompt_template: str = None,
) -> str:
    """Generate LLM response based on query and context."""
    try:
        args = {
            "question": query,
            "context": context,
        }
        user_prompt = render_prompt(user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE, args)
        system_prompt = read_query_prompt(system_prompt_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE)

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
    query: str,
    extractor: BaseEntityExtractor,
    context_provider: BaseContextProvider,
    system_prompt_template: str = None,
    user_prompt_template: str = None,
) -> List[str]:
    """Execute entity-based completion using provided components."""
    if not query or not isinstance(query, str):
        logger.error("Invalid query type or empty query")
        return ["Invalid query input"]

    try:
        logger.info(f"Processing query: {query[:100]}")

        entities = await extractor.extract_entities(query)
        if not entities:
            logger.info("No entities extracted")
            return ["No entities found"]

        context = await context_provider.get_context(entities, query)
        if not context:
            logger.info("No context retrieved")
            return ["No context found"]

        response = await get_llm_response(
            query, context, system_prompt_template, user_prompt_template
        )
        return [response]

    except Exception as e:
        logger.error(f"Entity completion failed: {str(e)}")
        return ["Entity completion failed"]


if __name__ == "__main__":
    # For testing purposes, will be removed by the end of the sprint
    import asyncio
    import logging
    from cognee.tasks.entity_completion.entity_extractors.dummy_entity_extractor import (
        DummyEntityExtractor,
    )
    from cognee.tasks.entity_completion.context_providers.dummy_context_provider import (
        DummyContextProvider,
    )

    logging.basicConfig(level=logging.INFO)

    async def run_entity_completion():
        # Uses config defaults
        result = await entity_completion(
            "Tell me about Einstein",
            DummyEntityExtractor(),
            DummyContextProvider(),
        )
        print(f"Query Response: {result[0]}")

    asyncio.run(run_entity_completion())

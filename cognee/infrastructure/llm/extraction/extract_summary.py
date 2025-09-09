from cognee.shared.logging_utils import get_logger
import os
from typing import Type

from instructor.core import InstructorRetryException
from cognee.infrastructure.llm.prompts import read_query_prompt
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.data_models import SummarizedCode

logger = get_logger("extract_summary")


def get_mock_summarized_code():
    """Local mock function to avoid circular imports."""
    return SummarizedCode(
        high_level_summary="Mock code summary",
        key_features=["Mock feature 1", "Mock feature 2"],
        imports=["mock_import"],
        constants=["MOCK_CONSTANT"],
        classes=[],
        functions=[],
        workflow_description="Mock workflow description",
    )


async def extract_summary(content: str, response_model: Type[BaseModel]):
    system_prompt = read_query_prompt("summarize_content.txt")

    llm_output = await LLMGateway.acreate_structured_output(content, system_prompt, response_model)

    return llm_output


async def extract_code_summary(content: str):
    enable_mocking = os.getenv("MOCK_CODE_SUMMARY", "false")
    if isinstance(enable_mocking, bool):
        enable_mocking = str(enable_mocking).lower()
    enable_mocking = enable_mocking in ("true", "1", "yes")

    if enable_mocking:
        result = get_mock_summarized_code()
        return result
    else:
        try:
            result = await extract_summary(content, response_model=SummarizedCode)
        except InstructorRetryException as e:
            logger.error("Failed to extract code summary, falling back to mock summary", exc_info=e)
            result = get_mock_summarized_code()

        return result

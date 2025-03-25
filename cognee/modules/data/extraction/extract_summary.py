from cognee.shared.logging_utils import get_logger
import os
from typing import Type

from instructor.exceptions import InstructorRetryException
from pydantic import BaseModel

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.shared.data_models import SummarizedCode
from cognee.tasks.summarization.mock_summary import get_mock_summarized_code

logger = get_logger("extract_summary")


async def extract_summary(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = read_query_prompt("summarize_content.txt")

    llm_output = await llm_client.acreate_structured_output(content, system_prompt, response_model)

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

import os

from instructor.core import InstructorRetryException
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.shared.data_models import SummarizedCode
from cognee.shared.logging_utils import get_logger

logger = get_logger("extract_summary")

SUMMARY_PROMPT_FILE = "summarize_content.txt"


def get_mock_summarized_code() -> SummarizedCode:
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


async def extract_summary_with_provenance(
    content: str, response_model: type[BaseModel]
) -> tuple[BaseModel, str, str]:
    """``extract_summary`` plus what eval capture records about the call (SDK-529).

    Returns ``(llm_output, prompt_text, model_name)``. The prompt text is the
    system prompt that was sent — already read for the call, so returning it
    costs nothing extra — and the model name is the model id the active LLM
    context config routes the call to (inside ``pipeline_stage("summarization")``
    that is the summarization-stage model; the same id the session usage
    tracker records). Used by ``summarize_text``; ``extract_summary`` keeps its
    single-value contract for every other caller.
    """
    system_prompt = read_query_prompt(SUMMARY_PROMPT_FILE) or ""
    model_name = get_llm_context_config().llm_model

    llm_output = await LLMGateway.acreate_structured_output(content, system_prompt, response_model)

    return llm_output, system_prompt, model_name


async def extract_summary(content: str, response_model: type[BaseModel]):
    llm_output, _prompt_text, _model_name = await extract_summary_with_provenance(
        content, response_model
    )

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

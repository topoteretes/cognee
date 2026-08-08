import os

from pydantic import BaseModel
from cognee.infrastructure.llm import get_llm_config


class LLMConfig(BaseModel):
    api_key: str
    model: str
    provider: str


async def save_llm_config(new_llm_config: LLMConfig):
    llm_config = get_llm_config()

    llm_config.llm_provider = new_llm_config.provider
    llm_config.llm_model = new_llm_config.model

    if "*****" not in new_llm_config.api_key and len(new_llm_config.api_key.strip()) > 0:
        llm_config.llm_api_key = new_llm_config.api_key.strip()
        # Export to the process environment as well, so code paths (and
        # subprocesses) that read LLM_API_KEY directly pick up the new key.
        os.environ["LLM_API_KEY"] = llm_config.llm_api_key

import json
import logging
from pydantic import BaseModel
from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure import infrastructure_config

class LLMConfig(BaseModel):
    apiKey: str
    model: str
    provider: str

async def save_llm_config(new_llm_config: LLMConfig):
    llm_config = get_llm_config()

    llm_config.llm_provider = new_llm_config.provider
    llm_config.llm_model = new_llm_config.model

    if "*****" not in new_llm_config.apiKey and len(new_llm_config.apiKey.strip()) > 0:
        llm_config.llm_api_key = new_llm_config.apiKey

    logging.error(json.dumps(llm_config.to_dict()))
    infrastructure_config.llm_engine = None

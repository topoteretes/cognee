import os
from pydantic import BaseModel
from cognee.config import Config

config = Config()

class LLMConfig(BaseModel):
    openAIApiKey: str

async def save_llm_config(llm_config: LLMConfig):
    if "*" in llm_config.openAIApiKey:
        return

    os.environ["OPENAI_API_KEY"] = llm_config.openAIApiKey
    config.load()

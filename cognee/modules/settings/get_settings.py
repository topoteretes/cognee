from enum import Enum
from pydantic import BaseModel
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.llm import get_llm_config


class ConfigChoice(BaseModel):
    value: str
    label: str


class ModelName(Enum):
    openai = "openai"
    ollama = "ollama"
    anthropic = "anthropic"
    gemini = "gemini"


class LLMConfig(BaseModel):
    api_key: str
    model: str
    provider: str
    endpoint: str
    api_version: str
    models: dict[str, list[ConfigChoice]]
    providers: list[ConfigChoice]


class VectorDBConfig(BaseModel):
    api_key: str
    url: str
    provider: str
    providers: list[ConfigChoice]


class SettingsDict(BaseModel):
    llm: LLMConfig
    vector_db: VectorDBConfig


def get_settings() -> SettingsDict:
    llm_config = get_llm_config()

    vector_dbs = [
        {
            "value": "weaviate",
            "label": "Weaviate",
        },
        {
            "value": "qdrant",
            "label": "Qdrant",
        },
        {
            "value": "lancedb",
            "label": "LanceDB",
        },
        {
            "value": "pgvector",
            "label": "PGVector",
        },
    ]

    vector_config = get_vectordb_config()

    llm_providers = [
        {
            "value": "openai",
            "label": "OpenAI",
        },
        {
            "value": "ollama",
            "label": "Ollama",
        },
        {
            "value": "anthropic",
            "label": "Anthropic",
        },
        {
            "value": "gemini",
            "label": "Gemini",
        },
    ]

    return SettingsDict.model_validate(
        dict(
            llm={
                "provider": llm_config.llm_provider,
                "model": llm_config.llm_model,
                "endpoint": llm_config.llm_endpoint,
                "api_version": llm_config.llm_api_version,
                "api_key": (llm_config.llm_api_key[0:10] + "*" * (len(llm_config.llm_api_key) - 10))
                if llm_config.llm_api_key
                else None,
                "providers": llm_providers,
                "models": {
                    "openai": [
                        {
                            "value": "gpt-4o-mini",
                            "label": "gpt-4o-mini",
                        },
                        {
                            "value": "gpt-4o",
                            "label": "gpt-4o",
                        },
                        {
                            "value": "gpt-4-turbo",
                            "label": "gpt-4-turbo",
                        },
                        {
                            "value": "gpt-3.5-turbo",
                            "label": "gpt-3.5-turbo",
                        },
                    ],
                    "ollama": [
                        {
                            "value": "llama3",
                            "label": "llama3",
                        },
                        {
                            "value": "mistral",
                            "label": "mistral",
                        },
                    ],
                    "anthropic": [
                        {
                            "value": "Claude 3 Opus",
                            "label": "Claude 3 Opus",
                        },
                        {
                            "value": "Claude 3 Sonnet",
                            "label": "Claude 3 Sonnet",
                        },
                        {
                            "value": "Claude 3 Haiku",
                            "label": "Claude 3 Haiku",
                        },
                    ],
                    "gemini": [
                        {
                            "value": "gemini-2.0-flash-exp",
                            "label": "Gemini 2.0 Flash",
                        },
                    ],
                },
            },
            vector_db={
                "provider": vector_config.vector_db_provider,
                "url": vector_config.vector_db_url,
                "api_key": (
                    vector_config.vector_db_key[0:10]
                    + "*" * (len(vector_config.vector_db_key) - 10)
                ),
                "providers": vector_dbs,
            },
        )
    )

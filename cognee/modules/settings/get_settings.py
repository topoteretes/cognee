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


class LLMConfig(BaseModel):
    api_key: str
    model: ConfigChoice
    provider: ConfigChoice
    models: dict[str, list[ConfigChoice]]
    providers: list[ConfigChoice]


class VectorDBConfig(BaseModel):
    api_key: str
    url: str
    provider: ConfigChoice
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
    ]

    return SettingsDict.model_validate(
        dict(
            llm={
                "provider": {
                    "label": llm_config.llm_provider,
                    "value": llm_config.llm_provider,
                }
                if llm_config.llm_provider
                else llm_providers[0],
                "model": {
                    "value": llm_config.llm_model,
                    "label": llm_config.llm_model,
                }
                if llm_config.llm_model
                else None,
                "api_key": (llm_config.llm_api_key[:-10] + "**********")
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
                },
            },
            vector_db={
                "provider": {
                    "label": vector_config.vector_db_provider,
                    "value": vector_config.vector_db_provider.lower(),
                },
                "url": vector_config.vector_db_url,
                "api_key": vector_config.vector_db_key,
                "providers": vector_dbs,
            },
        )
    )

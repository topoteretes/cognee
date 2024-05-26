from cognee.config import Config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.llm import get_llm_config

def get_settings():
    config = Config()
    config.load()
    llm_config = get_llm_config()

    vector_dbs = [{
        "value": "weaviate",
        "label": "Weaviate",
    }, {
        "value": "qdrant",
        "label": "Qdrant",
    }, {
        "value": "lancedb",
        "label": "LanceDB",
    }]

    vector_config = get_vectordb_config()

    llm_providers = [{
        "value": "openai",
        "label": "OpenAI",
    }, {
        "value": "ollama",
        "label": "Ollama",
    }, {
        "value": "anthropic",
        "label": "Anthropic",
    }]

    llm_config = get_llm_config()

    return dict(
        llm = {
            "provider": {
                "label": llm_config.llm_provider,
                "value": llm_config.llm_provider,
            } if llm_config.llm_provider else llm_providers[0],
            "model": {
                "value": llm_config.llm_model,
                "label": llm_config.llm_model,
            } if llm_config.llm_model else None,
            "apiKey": llm_config.llm_api_key[:-10] + "**********" if llm_config.llm_api_key else None,
            "providers": llm_providers,
            "models": {
                "openai": [{
                    "value": "gpt-4o",
                    "label": "gpt-4o",
                }, {
                    "value": "gpt-4-turbo",
                    "label": "gpt-4-turbo",
                }, {
                    "value": "gpt-3.5-turbo",
                    "label": "gpt-3.5-turbo",
                }],
                "ollama": [{
                    "value": "llama3",
                    "label": "llama3",
                }, {
                    "value": "mistral",
                    "label": "mistral",
                }],
                "anthropic": [{
                    "value": "Claude 3 Opus",
                    "label": "Claude 3 Opus",
                }, {
                    "value": "Claude 3 Sonnet",
                    "label": "Claude 3 Sonnet",
                }, {
                    "value": "Claude 3 Haiku",
                    "label": "Claude 3 Haiku",
                }]
            },
        },
        vectorDB = {
            "provider": {
                "label": vector_config.vector_engine_provider,
                "value": vector_config.vector_engine_provider.lower(),
            },
            "url": vector_config.vector_db_url,
            "apiKey": vector_config.vector_db_key,
            "options": vector_dbs,
        },
    )

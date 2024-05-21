from cognee.config import Config
from cognee.infrastructure import infrastructure_config

config = Config()
config.load()

def get_settings():
    vector_engine_choice = infrastructure_config.get_config()["vector_engine_choice"]
    vector_db_options = [{
      "value": "weaviate",
      "label": "Weaviate",
    }, {
      "value": "qdrant",
      "label": "Qdrant",
    }, {
      "value": "lancedb",
      "label": "LanceDB",
    }]

    vector_db_config = dict(
      url = config.weaviate_url,
      apiKey = config.weaviate_api_key,
      choice = vector_db_options[0],
      options = vector_db_options,
    ) if vector_engine_choice == "weaviate" else dict(
      url = config.qdrant_url,
      apiKey = config.qdrant_api_key,
      choice = vector_db_options[1],
      options = vector_db_options,
    ) if vector_engine_choice == "qdrant" else dict(
      url = infrastructure_config.get_config("lance_db_path"),
      choice = vector_db_options[2],
      options = vector_db_options,
    )

    return dict(
        llm = dict(
          openAIApiKey = config.openai_key[:-10] + "**********",
        ),
        vectorDB = vector_db_config,
    )

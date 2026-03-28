"""This module is used to set the configuration of the system."""

import os
from cognee.base_config import get_base_config
from cognee.modules.cognify.config import get_cognify_config
from cognee.infrastructure.data.chunking.config import get_chunk_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.llm.config import (
    get_llm_config,
)
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.databases.relational import get_relational_config, get_migration_config
from cognee.tasks.translation.config import get_translation_config
from cognee.api.v1.exceptions.exceptions import InvalidConfigAttributeError


class config:
    """
    Configuration namespace for Cognee's runtime settings.

    All methods are static and configure LLM providers, embedding providers,
    database backends, chunking strategies, and storage paths at runtime
    without requiring environment variable changes.

    Example:
        ```python
        import cognee

        cognee.config.set_llm_api_key("your-api-key")
        cognee.config.set_llm_model("gpt-4o-mini")
        cognee.config.set_embedding_provider("fastembed")
        cognee.config.set_embedding_model("BAAI/bge-small-en-v1.5")
        cognee.config.set_embedding_dimensions(384)
        cognee.config.system_root_directory("/path/to/system")
        cognee.config.set_vector_db_provider("chromadb")
        ```
    """

    @staticmethod
    def system_root_directory(system_root_directory: str):
        """Set the system root directory and update dependent database paths.

        Configures the base system directory and cascades the change to
        relational, graph, and vector database path configurations.

        Parameters
        ----------
        system_root_directory : str
            Absolute path to the system root directory.
        """
        base_config = get_base_config()
        base_config.system_root_directory = system_root_directory

        databases_directory_path = os.path.join(base_config.system_root_directory, "databases")

        relational_config = get_relational_config()
        relational_config.db_path = databases_directory_path

        graph_config = get_graph_config()
        graph_file_name = graph_config.graph_filename
        graph_config.graph_file_path = os.path.join(databases_directory_path, graph_file_name)

        vector_config = get_vectordb_config()
        if vector_config.vector_db_provider == "lancedb":
            vector_config.vector_db_url = os.path.join(databases_directory_path, "cognee.lancedb")

    @staticmethod
    def data_root_directory(data_root_directory: str):
        """Set the data root directory for storing ingested data.

        Parameters
        ----------
        data_root_directory : str
            Absolute path to the data root directory.
        """
        base_config = get_base_config()
        base_config.data_root_directory = data_root_directory

    @staticmethod
    def monitoring_tool(monitoring_tool: object):
        """Set the monitoring tool for observability.

        Parameters
        ----------
        monitoring_tool : object
            A monitoring tool instance.
        """
        base_config = get_base_config()
        base_config.monitoring_tool = monitoring_tool

    @staticmethod
    def set_classification_model(classification_model: object):
        """Set the model used for classification during cognification.

        Parameters
        ----------
        classification_model : object
            The classification model to use.
        """
        cognify_config = get_cognify_config()
        cognify_config.classification_model = classification_model

    @staticmethod
    def set_summarization_model(summarization_model: object):
        """Set the model used for summarization during cognification.

        Parameters
        ----------
        summarization_model : object
            The summarization model to use.
        """
        cognify_config = get_cognify_config()
        cognify_config.summarization_model = summarization_model

    @staticmethod
    def set_graph_model(graph_model: object):
        """Set the model used for graph extraction.

        Parameters
        ----------
        graph_model : object
            The graph extraction model to use.
        """
        graph_config = get_graph_config()
        graph_config.graph_model = graph_model

    @staticmethod
    def set_graph_database_provider(graph_database_provider: str):
        """Set the graph database provider.

        Parameters
        ----------
        graph_database_provider : str
            The graph database provider name (e.g. 'networkx', 'neo4j').
        """
        graph_config = get_graph_config()
        graph_config.graph_database_provider = graph_database_provider

    @staticmethod
    def set_llm_provider(llm_provider: str):
        """Set the LLM provider.

        Parameters
        ----------
        llm_provider : str
            The LLM provider name (e.g. 'openai', 'anthropic', 'litellm').
        """
        llm_config = get_llm_config()
        llm_config.llm_provider = llm_provider

    @staticmethod
    def set_llm_endpoint(llm_endpoint: str):
        """Set a custom LLM API endpoint URL.

        Parameters
        ----------
        llm_endpoint : str
            The base URL for the LLM API.
        """
        llm_config = get_llm_config()
        llm_config.llm_endpoint = llm_endpoint

    @staticmethod
    def set_llm_model(llm_model: str):
        """Set the LLM model name.

        Parameters
        ----------
        llm_model : str
            The model identifier (e.g. 'gpt-4o-mini', 'claude-3-sonnet').
        """
        llm_config = get_llm_config()
        llm_config.llm_model = llm_model

    @staticmethod
    def set_llm_api_key(llm_api_key: str):
        """Set the API key for the LLM provider.

        Parameters
        ----------
        llm_api_key : str
            The API key string.
        """
        llm_config = get_llm_config()
        llm_config.llm_api_key = llm_api_key

    @staticmethod
    def _update_config(config_obj, config_dict: dict):
        """Update a config object with values from a dictionary after attribute validation.

        Parameters
        ----------
        config_obj : object
            The configuration object to update.
        config_dict : dict
            A dictionary of attribute names to new values.

        Returns
        -------
        object
            The updated configuration object.

        Raises
        ------
        InvalidConfigAttributeError
            If any key in config_dict is not a valid attribute of config_obj.
        """
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                object.__setattr__(config_obj, key, value)
            else:
                raise InvalidConfigAttributeError(attribute=key)

        return config_obj

    @staticmethod
    def set_llm_config(config_dict: dict):
        """Update the LLM config with values from a dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary of LLM config attributes to update.
        """
        config._update_config(get_llm_config(), config_dict)

    # Embedding configuration methods

    @staticmethod
    def set_embedding_provider(embedding_provider: str):
        """Set the embedding provider.

        Parameters
        ----------
        embedding_provider : str
            The embedding provider name (e.g. 'openai', 'fastembed', 'azure', 'litellm').
        """
        embedding_config = get_embedding_config()
        embedding_config.embedding_provider = embedding_provider

    @staticmethod
    def set_embedding_model(embedding_model: str):
        """Set the embedding model name.

        Parameters
        ----------
        embedding_model : str
            The model identifier (e.g. 'openai/text-embedding-3-large', 'BAAI/bge-small-en-v1.5').
        """
        embedding_config = get_embedding_config()
        embedding_config.embedding_model = embedding_model

    @staticmethod
    def set_embedding_dimensions(embedding_dimensions: int):
        """Set the embedding vector dimensions.

        Coerces string inputs to int for CLI compatibility and validates
        the value is a positive integer.

        Parameters
        ----------
        embedding_dimensions : int
            The number of dimensions for embedding vectors
            (e.g. 3072 for text-embedding-3-large, 384 for bge-small-en-v1.5).

        Raises
        ------
        ValueError
            If the value cannot be converted to an integer or is not positive.
        """
        if isinstance(embedding_dimensions, str):
            try:
                embedding_dimensions = int(embedding_dimensions)
            except ValueError as exc:
                raise ValueError("embedding_dimensions must be a positive integer.") from exc
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be a positive integer.")
        embedding_config = get_embedding_config()
        embedding_config.embedding_dimensions = embedding_dimensions

    @staticmethod
    def set_embedding_endpoint(embedding_endpoint: str):
        """Set a custom embedding API endpoint URL.

        Parameters
        ----------
        embedding_endpoint : str
            The base URL for the embedding API.
        """
        embedding_config = get_embedding_config()
        embedding_config.embedding_endpoint = embedding_endpoint

    @staticmethod
    def set_embedding_api_key(embedding_api_key: str):
        """Set the API key for the embedding provider.

        Parameters
        ----------
        embedding_api_key : str
            The API key string.
        """
        embedding_config = get_embedding_config()
        embedding_config.embedding_api_key = embedding_api_key

    @staticmethod
    def set_embedding_config(config_dict: dict):
        """Update the embedding config with values from a dictionary.

        Routes embedding_dimensions through set_embedding_dimensions to ensure
        type coercion and positive-value validation are applied.

        Parameters
        ----------
        config_dict : dict
            A dictionary of embedding config attributes to update.
            Valid keys include: embedding_provider, embedding_model,
            embedding_dimensions, embedding_endpoint, embedding_api_key,
            embedding_api_version, embedding_max_completion_tokens,
            embedding_batch_size, huggingface_tokenizer.

        Example
        -------
            ```python
            cognee.config.set_embedding_config({
                "embedding_provider": "fastembed",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_dimensions": 384,
            })
            ```
        """
        normalized_config = dict(config_dict)
        if "embedding_dimensions" in normalized_config:
            config.set_embedding_dimensions(normalized_config.pop("embedding_dimensions"))
        config._update_config(get_embedding_config(), normalized_config)

    @staticmethod
    def set_chunk_strategy(chunk_strategy: object):
        """Set the chunking strategy.

        Parameters
        ----------
        chunk_strategy : object
            The chunking strategy to use.
        """
        chunk_config = get_chunk_config()
        chunk_config.chunk_strategy = chunk_strategy

    @staticmethod
    def set_chunk_engine(chunk_engine: object):
        """Set the chunking engine.

        Parameters
        ----------
        chunk_engine : object
            The chunking engine to use.
        """
        chunk_config = get_chunk_config()
        chunk_config.chunk_engine = chunk_engine

    @staticmethod
    def set_chunk_overlap(chunk_overlap: object):
        """Set the chunk overlap size.

        Parameters
        ----------
        chunk_overlap : object
            The number of overlapping tokens/characters between chunks.
        """
        chunk_config = get_chunk_config()
        chunk_config.chunk_overlap = chunk_overlap

    @staticmethod
    def set_chunk_size(chunk_size: object):
        """Set the chunk size.

        Parameters
        ----------
        chunk_size : object
            The target size for each chunk.
        """
        chunk_config = get_chunk_config()
        chunk_config.chunk_size = chunk_size

    @staticmethod
    def set_vector_db_provider(vector_db_provider: str):
        """Set the vector database provider.

        Parameters
        ----------
        vector_db_provider : str
            The vector database provider name (e.g. 'lancedb', 'chromadb', 'qdrant').
        """
        vector_db_config = get_vectordb_config()
        vector_db_config.vector_db_provider = vector_db_provider

    @staticmethod
    def set_relational_db_config(config_dict: dict):
        """Update the relational database config with values from a dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary of relational DB config attributes to update.
        """
        config._update_config(get_relational_config(), config_dict)

    @staticmethod
    def set_migration_db_config(config_dict: dict):
        """Update the migration database config with values from a dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary of migration DB config attributes to update.
        """
        config._update_config(get_migration_config(), config_dict)

    @staticmethod
    def set_graph_db_config(config_dict: dict) -> None:
        """Update the graph database config with values from a dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary of graph DB config attributes to update.
        """
        config._update_config(get_graph_config(), config_dict)

    @staticmethod
    def set_vector_db_config(config_dict: dict):
        """Update the vector database config with values from a dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary of vector DB config attributes to update.
        """
        config._update_config(get_vectordb_config(), config_dict)

    @staticmethod
    def set_vector_db_key(db_key: str):
        """Set the API key for the vector database provider.

        Parameters
        ----------
        db_key : str
            The API key string.
        """
        vector_db_config = get_vectordb_config()
        vector_db_config.vector_db_key = db_key

    @staticmethod
    def set_vector_db_url(db_url: str):
        """Set the URL for the vector database.

        Parameters
        ----------
        db_url : str
            The database URL.
        """
        vector_db_config = get_vectordb_config()
        vector_db_config.vector_db_url = db_url

    # Translation configuration methods

    @staticmethod
    def set_translation_provider(provider: str):
        """Set the translation provider.

        Parameters
        ----------
        provider : str
            The translation provider name (e.g. 'llm', 'google', 'azure').
        """
        translation_config = get_translation_config()
        translation_config.translation_provider = provider

    @staticmethod
    def set_translation_target_language(target_language: str):
        """Set the default target language for translations.

        Parameters
        ----------
        target_language : str
            The target language code (e.g. 'en', 'es', 'fr').
        """
        translation_config = get_translation_config()
        translation_config.target_language = target_language

    @staticmethod
    def set_translation_config(config_dict: dict):
        """Update the translation config with values from a dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary of translation config attributes to update.
        """
        config._update_config(get_translation_config(), config_dict)

    @staticmethod
    def set(key: str, value):
        """Set a configuration value by key name.

        Generic setter that maps configuration keys to their specific setter methods.
        This enables CLI commands like 'cognee config set llm_api_key <value>'.

        For embedding keys not explicitly listed in the mapping but present on
        EmbeddingConfig, the value is routed through set_embedding_config as a
        fallback so that all valid embedding attributes are accessible via CLI.

        Parameters
        ----------
        key : str
            The configuration key name.
        value : any
            The value to set.

        Raises
        ------
        InvalidConfigAttributeError
            If the key is not a recognized configuration attribute.
        """
        # Map configuration keys to their setter methods
        setter_mapping = {
            "llm_provider": "set_llm_provider",
            "llm_model": "set_llm_model",
            "llm_api_key": "set_llm_api_key",
            "llm_endpoint": "set_llm_endpoint",
            "embedding_provider": "set_embedding_provider",
            "embedding_model": "set_embedding_model",
            "embedding_dimensions": "set_embedding_dimensions",
            "embedding_endpoint": "set_embedding_endpoint",
            "embedding_api_key": "set_embedding_api_key",
            "graph_database_provider": "set_graph_database_provider",
            "vector_db_provider": "set_vector_db_provider",
            "vector_db_url": "set_vector_db_url",
            "vector_db_key": "set_vector_db_key",
            "chunk_size": "set_chunk_size",
            "chunk_overlap": "set_chunk_overlap",
            "chunk_strategy": "set_chunk_strategy",
            "chunk_engine": "set_chunk_engine",
            "classification_model": "set_classification_model",
            "summarization_model": "set_summarization_model",
            "graph_model": "set_graph_model",
            "system_root_directory": "system_root_directory",
            "data_root_directory": "data_root_directory",
        }

        if key in setter_mapping:
            method_name = setter_mapping[key]
            method = getattr(config, method_name)
            method(value)
            return

        embedding_config = get_embedding_config()
        if hasattr(embedding_config, key):
            config.set_embedding_config({key: value})
            return

        raise InvalidConfigAttributeError(attribute=key)

"""Configuration for cognee - cognitive architecture framework."""
import logging
import os
import configparser
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from cognee.root_dir import get_absolute_path
from cognee.shared.data_models import ChunkStrategy

base_dir = Path(__file__).resolve().parent.parent
# Load the .env file from the base directory
dotenv_path = base_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)


@dataclass
class Config:
    """ Configuration for cognee - cognitive architecture framework. """
    cognee_dir: str = field(
        default_factory=lambda: os.getenv("COG_ARCH_DIR", "cognee")
    )
    config_path: str = field(
        default_factory=lambda: os.path.join(
            os.getenv("COG_ARCH_DIR", "cognee"), "config"
        )
    )


    system_root_directory = get_absolute_path(".cognee_system")
    data_root_directory = os.getenv("DATA_PATH", get_absolute_path(".data"))

    vectordb: str = os.getenv("VECTORDB", "weaviate")

    qdrant_path: str = os.getenv("QDRANT_PATH", None)
    qdrant_url: str = os.getenv("QDRANT_URL", None)
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", None)

    db_path = str = os.getenv("COGNEE_DB_PATH", "databases")
    db_name: str = os.getenv("DB_NAME", "cognee.db")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_user: str = os.getenv("DB_USER", "cognee")
    db_password: str = os.getenv("DB_PASSWORD", "cognee")

    sqlalchemy_logging: bool = os.getenv("SQLALCHEMY_LOGGING", True)

    graph_filename = os.getenv("GRAPH_NAME", "cognee_graph.pkl")

    # Model parameters
    llm_provider: str = os.getenv("LLM_PROVIDER","openai") #openai, or custom or ollama
    custom_model: str = os.getenv("CUSTOM_LLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1") #"mistralai/Mixtral-8x7B-Instruct-v0.1"
    custom_endpoint: str = os.getenv("CUSTOM_ENDPOINT", "https://api.endpoints.anyscale.com/v1") #"https://api.endpoints.anyscale.com/v1" # pass claude endpoint
    custom_key: Optional[str] = os.getenv("CUSTOM_LLM_API_KEY")
    ollama_endpoint: str = os.getenv("CUSTOM_OLLAMA_ENDPOINT", "http://localhost:11434/v1") #"http://localhost:11434/v1"
    ollama_key: Optional[str] = "ollama"
    ollama_model: str = os.getenv("CUSTOM_OLLAMA_MODEL", "mistral:instruct") #"mistral:instruct"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-1106-preview" ) #"gpt-4-1106-preview"
    model_endpoint: str = "openai"
    openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    openai_embedding_model = "text-embedding-3-large"
    openai_embedding_dimensions = 3072

    graphistry_username = os.getenv("GRAPHISTRY_USERNAME")
    graphistry_password = os.getenv("GRAPHISTRY_PASSWORD")

    # Embedding parameters
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimensions: int = 1024
    connect_documents: bool = False

    # Database parameters
    graph_database_provider: str = os.getenv("GRAPH_DB_PROVIDER", "NETWORKX")

    if (
        os.getenv("ENV") == "prod"
        or os.getenv("ENV") == "dev"
        or os.getenv("AWS_ENV") == "dev"
        or os.getenv("AWS_ENV") == "prd"
    ):
        load_dotenv()
        logging.info("graph_db_url: %s", os.getenv("GRAPH_DB_URL_PROD"))
        graph_database_url: str = os.getenv("GRAPH_DB_URL_PROD")
        graph_database_username: str = os.getenv("GRAPH_DB_USER")
        graph_database_password: str = os.getenv("GRAPH_DB_PW")
    else:
        logging.info("graph_db_url: %s", os.getenv("GRAPH_DB_URL"))
        graph_database_url: str = os.getenv("GRAPH_DB_URL")
        graph_database_username: str = os.getenv("GRAPH_DB_USER")
        graph_database_password: str = os.getenv("GRAPH_DB_PW")

    weaviate_url: str = os.getenv("WEAVIATE_URL")
    weaviate_api_key: str = os.getenv("WEAVIATE_API_KEY")

    if (
        os.getenv("ENV") == "prod"
        or os.getenv("ENV") == "dev"
        or os.getenv("AWS_ENV") == "dev"
        or os.getenv("AWS_ENV") == "prd"
    ):
        load_dotenv()

        db_host: str = os.getenv("POSTGRES_HOST")
        logging.info("db_host: %s", db_host)
        db_user: str = os.getenv("POSTGRES_USER")
        db_password: str = os.getenv("POSTGRES_PASSWORD")
        db_name: str = os.getenv("POSTGRES_DB")

    # Model parameters and configuration for interlayer scoring
    intra_layer_score_treshold: float = 0.98


    # Client ID
    anon_clientid: Optional[str] = field(default_factory=lambda: uuid.uuid4().hex)

    #Chunking parameters
    chunk_size: int = 1500
    chunk_overlap: int = 0
    chunk_strategy: str = ChunkStrategy.PARAGRAPH

    def load(self):
        """Loads the configuration from a file or environment variables."""
        config = configparser.ConfigParser()
        config.read(self.config_path)

        # Override with environment variables if they exist
        for attr in self.__annotations__:
            env_value = os.getenv(attr.upper())
            if env_value is not None:
                setattr(self, attr, env_value)

        # Load from config file
        if config.sections():
            for section in config.sections():
                for key, value in config.items(section):
                    if hasattr(self, key):
                        setattr(self, key, value)

    def save(self):
        """Saves the current configuration to a file."""
        config = configparser.ConfigParser()

        # Save the current settings to the config file
        for attr, value in self.__dict__.items():
            section, option = attr.split("_", 1)
            if not config.has_section(section):
                config.add_section(section)
            config.set(section, option, str(value))

        with open(self.config_path, "w") as configfile:
            config.write(configfile)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the configuration."""
        return {attr: getattr(self, attr) for attr in self.__annotations__}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Creates a Config instance from a dictionary."""
        config = cls()
        for attr, value in config_dict.items():
            if hasattr(config, attr):
                setattr(config, attr, value)
        return config
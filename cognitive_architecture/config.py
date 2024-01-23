import os
import json
import configparser
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


base_dir = Path(__file__).resolve().parent.parent
# Load the .env file from the base directory
dotenv_path = base_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

@dataclass
class Config:
    # Paths and Directories
    memgpt_dir: str = field(default_factory=lambda: os.getenv('COG_ARCH_DIR', 'cognitive_achitecture'))
    config_path: str = field(default_factory=lambda: os.path.join(os.getenv('COG_ARCH_DIR', 'cognitive_achitecture'), 'config'))

    # Model parameters
    model: str = 'gpt-4-1106-preview'
    model_endpoint: str = 'openai'
    openai_key: Optional[str] = os.getenv('OPENAI_API_KEY')
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", 0.0))

    # Embedding parameters
    embedding_model: str = 'openai'
    embedding_dim: int = 1536
    embedding_chunk_size: int = 300

    # Database parameters
    if os.getenv('ENV') == 'prod' or os.getenv('ENV') == 'dev' or os.getenv('AWS_ENV') == 'dev' or   os.getenv('AWS_ENV') == 'prd':
        graph_database_url: str = os.getenv('GRAPH_DB_URL_PROD')
        graph_database_username: str = os.getenv('GRAPH_DB_USER')
        graph_database_password: str = os.getenv('GRAPH_DB_PW')
    else:
        graph_database_url: str = os.getenv('GRAPH_DB_URL')
        graph_database_username: str = os.getenv('GRAPH_DB_USER')
        graph_database_password: str = os.getenv('GRAPH_DB_PW')
    weaviate_url: str = os.getenv('WEAVIATE_URL')
    weaviate_api_key: str = os.getenv('WEAVIATE_API_KEY')
    postgres_user: str = os.getenv('POSTGRES_USER')
    postgres_password: str = os.getenv('POSTGRES_PASSWORD')
    postgres_db: str = os.getenv('POSTGRES_DB')
    if os.getenv('ENV') == 'prod' or os.getenv('ENV') == 'dev' or os.getenv('AWS_ENV') == 'dev' or os.getenv('AWS_ENV') == 'prd':
        postgres_host: str = os.getenv('POSTGRES_PROD_HOST')
    elif os.getenv('ENV') == 'docker':
        postgres_host: str = os.getenv('POSTGRES_HOST_DOCKER')
    elif os.getenv('ENV') == 'local':
        postgres_host: str = os.getenv('POSTGRES_HOST_LOCAL')





    # Client ID
    anon_clientid: Optional[str] = field(default_factory=lambda: uuid.uuid4().hex)

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
            section, option = attr.split('_', 1)
            if not config.has_section(section):
                config.add_section(section)
            config.set(section, option, str(value))

        with open(self.config_path, 'w') as configfile:
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

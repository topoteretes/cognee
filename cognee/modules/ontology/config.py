"""Ontology configuration following Cognee patterns."""

import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology.config")


class OntologyConfig(BaseSettings):
    """
    Configuration settings for the ontology system.
    
    Follows Cognee's BaseSettings pattern with environment variable support.
    """
    
    # Default ontology settings
    default_format: str = "json"
    enable_semantic_search: bool = False
    registry_type: str = "memory"  # "memory" or "database"
    
    # Provider settings
    rdf_provider_enabled: bool = True
    json_provider_enabled: bool = True
    csv_provider_enabled: bool = True
    
    # Performance settings
    cache_ontologies: bool = True
    max_cache_size: int = 100
    similarity_threshold: float = 0.8
    
    # Domain-specific settings
    medical_domain_enabled: bool = True
    legal_domain_enabled: bool = True
    code_domain_enabled: bool = True
    
    # File paths
    ontology_data_directory: str = os.path.join(
        os.getenv("COGNEE_DATA_ROOT", ".data_storage"), "ontologies"
    )
    default_config_file: Optional[str] = None
    
    # Environment variables
    ontology_api_key: Optional[str] = os.getenv("ONTOLOGY_API_KEY")
    ontology_endpoint: Optional[str] = os.getenv("ONTOLOGY_ENDPOINT")
    
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary following Cognee pattern."""
        return {
            "default_format": self.default_format,
            "enable_semantic_search": self.enable_semantic_search,
            "registry_type": self.registry_type,
            "rdf_provider_enabled": self.rdf_provider_enabled,
            "json_provider_enabled": self.json_provider_enabled,
            "csv_provider_enabled": self.csv_provider_enabled,
            "cache_ontologies": self.cache_ontologies,
            "max_cache_size": self.max_cache_size,
            "similarity_threshold": self.similarity_threshold,
            "medical_domain_enabled": self.medical_domain_enabled,
            "legal_domain_enabled": self.legal_domain_enabled,
            "code_domain_enabled": self.code_domain_enabled,
            "ontology_data_directory": self.ontology_data_directory,
        }


@lru_cache
def get_ontology_config():
    """Get ontology configuration instance following Cognee pattern."""
    return OntologyConfig()


# Configuration helpers following existing patterns
def set_ontology_data_directory(directory: str):
    """Set ontology data directory."""
    config = get_ontology_config()
    config.ontology_data_directory = directory
    logger.info(f"Set ontology data directory to: {directory}")


def enable_semantic_search(enabled: bool = True):
    """Enable or disable semantic search."""
    config = get_ontology_config()
    config.enable_semantic_search = enabled
    logger.info(f"Semantic search {'enabled' if enabled else 'disabled'}")


def set_similarity_threshold(threshold: float):
    """Set similarity threshold for entity matching."""
    config = get_ontology_config()
    config.similarity_threshold = threshold
    logger.info(f"Set similarity threshold to: {threshold}")

"""This module contains the configuration for ontology handling."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class OntologyEnvConfig(BaseSettings):
    """
    Represents the configuration for ontology handling, including parameters for
    ontology file storage and resolution/matching strategies.

    Public methods:
    - to_dict

    Instance variables:
    - ontology_resolver
    - ontology_matching
    - ontology_file_path
    - ontology_mode
    - model_config
    """

    ontology_resolver: str = "rdflib"
    matching_strategy: str = "fuzzy"
    ontology_file_path: str = ""
    ontology_mode: Literal["annotate", "strict"] = "annotate"

    model_config = SettingsConfigDict(env_file=".env", extra="allow", populate_by_name=True)

    def to_dict(self) -> dict:
        """
        Return the configuration as a dictionary.
        """
        return {
            "ontology_resolver": self.ontology_resolver,
            "matching_strategy": self.matching_strategy,
            "ontology_file_path": self.ontology_file_path,
            "ontology_mode": self.ontology_mode,
        }


@lru_cache
def get_ontology_env_config():
    """
    Retrieve the ontology configuration. This function utilizes caching to return a
    singleton instance of the OntologyConfig class for efficiency.
    """
    return OntologyEnvConfig()

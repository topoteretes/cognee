"""This module contains the configuration for ontology handling."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology_env_config")

VALID_ONTOLOGY_MODES: frozenset[str] = frozenset({"annotate", "strict"})
DEFAULT_ONTOLOGY_MODE = "annotate"


def normalize_ontology_mode(mode: "str | None") -> str:
    """Normalize an ontology mode value, falling back to the default with a warning.

    An unknown value must never break a cognify run (the config is constructed even
    for runs with no ontology at all), so this warns and falls back instead of raising.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in VALID_ONTOLOGY_MODES:
        logger.warning(
            "Unknown ONTOLOGY_MODE=%r — falling back to %r. Valid values: %s",
            mode,
            DEFAULT_ONTOLOGY_MODE,
            ", ".join(sorted(VALID_ONTOLOGY_MODES)),
        )
        return DEFAULT_ONTOLOGY_MODE
    return normalized_mode


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
    ontology_mode: str = DEFAULT_ONTOLOGY_MODE

    model_config = SettingsConfigDict(env_file=".env", extra="allow", populate_by_name=True)

    @field_validator("ontology_mode", mode="before")
    @classmethod
    def _normalize_ontology_mode(cls, value) -> str:
        return normalize_ontology_mode(value)

    def to_dict(self) -> dict:
        """
        Return the resolver-factory keyword arguments as a dictionary.

        Note: this dict is splatted into ``get_ontology_resolver_from_env`` — it must
        contain exactly that function's parameters. ``ontology_mode`` deliberately does
        not belong here; it is read separately via ``get_configured_ontology_mode``.
        """
        return {
            "ontology_resolver": self.ontology_resolver,
            "matching_strategy": self.matching_strategy,
            "ontology_file_path": self.ontology_file_path,
        }


@lru_cache
def get_ontology_env_config():
    """
    Retrieve the ontology configuration. This function utilizes caching to return a
    singleton instance of the OntologyConfig class for efficiency.
    """
    return OntologyEnvConfig()

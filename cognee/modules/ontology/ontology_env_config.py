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


def parse_authoritative_sources(raw: "str | dict | None") -> dict:
    """Accept the env string (JSON object or ``table[=owner]`` csv) or an already-built dict.

    Never raises: a value that cannot be parsed is logged and treated as empty, so a
    typo in an env var cannot break ingestion.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if text.startswith("{"):
        import json

        try:
            parsed = json.loads(text)
        except ValueError:
            logger.warning("ONTOLOGY_AUTHORITATIVE_SOURCES is not valid JSON; ignoring it.")
            return {}
        return parsed if isinstance(parsed, dict) else {}
    sources: dict = {}
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        table, _, owner = entry.partition("=")
        sources[table.strip()] = {"authoritative": True, "owner": owner.strip() or None}
    return sources


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
    - ontology_query_grounding
    - ontology_authoritative_sources
    - model_config
    """

    ontology_resolver: str = "rdflib"
    matching_strategy: str = "fuzzy"
    ontology_file_path: str = ""
    ontology_mode: str = DEFAULT_ONTOLOGY_MODE
    # Read-time use of the ontology: graph-completion and hybrid searches resolve query
    # terms against the ontology, pin the matched nodes as seeds and tell the LLM what
    # each term means. Only active when ontology_file_path is set.
    ontology_query_grounding: bool = True
    # Which schema tables are the system of record for the concept they realize.
    # Either a JSON object ({"crm.customers": {"owner": "sales-ops"}, "orders": true})
    # or a comma list of ``table`` / ``table=owner`` entries. Per-call
    # ``ontology_config["authoritative_sources"]`` overrides it.
    ontology_authoritative_sources: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="allow", populate_by_name=True)

    @field_validator("ontology_mode", mode="before")
    @classmethod
    def _normalize_ontology_mode(cls, value) -> str:
        return normalize_ontology_mode(value)

    def authoritative_sources(self) -> dict:
        """Parse ``ontology_authoritative_sources`` into the per-call dict shape."""
        return parse_authoritative_sources(self.ontology_authoritative_sources)

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

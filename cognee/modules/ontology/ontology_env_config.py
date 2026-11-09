"""This module contains the configuration for ontology handling."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology_env_config")

VALID_ONTOLOGY_MODES: frozenset[str] = frozenset({"annotate", "strict"})
DEFAULT_ONTOLOGY_MODE = "annotate"

VALID_ONTOLOGY_REASONING: frozenset[str] = frozenset({"none", "owl_rl"})
DEFAULT_ONTOLOGY_REASONING = "none"


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


def parse_authoritative_sources(raw: str) -> dict[str, "str | None"]:
    """Parse ``ONTOLOGY_AUTHORITATIVE_SOURCES``: comma-separated ``table[=owner]`` entries.

    Returns table name (bare or ``schema.table``) to owner, None when no owner is
    given; every listed table is authoritative. An entry with no table name raises.
    """
    sources: dict[str, str | None] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        table, _, owner = entry.partition("=")
        if not table.strip():
            raise ValueError(f"ONTOLOGY_AUTHORITATIVE_SOURCES entry {entry!r} names no table")
        sources[table.strip()] = owner.strip() or None
    return sources


def normalize_ontology_reasoning(reasoning: "str | None") -> str:
    """Normalize an ontology reasoning value, falling back to the default with a warning.

    Mirrors ``normalize_ontology_mode``: an unknown value warns and falls back to
    ``none`` instead of raising, so a typo never breaks a cognify run.
    """
    normalized_reasoning = str(reasoning).strip().lower()
    if normalized_reasoning not in VALID_ONTOLOGY_REASONING:
        logger.warning(
            "Unknown ONTOLOGY_REASONING=%r — falling back to %r. Valid values: %s",
            reasoning,
            DEFAULT_ONTOLOGY_REASONING,
            ", ".join(sorted(VALID_ONTOLOGY_REASONING)),
        )
        return DEFAULT_ONTOLOGY_REASONING
    return normalized_reasoning


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
    - ontology_reasoning
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
    # Which schema tables are the system of record for the concept they realize: a comma
    # list of ``table`` / ``table=owner`` entries. Per-call
    # ``ontology_config["authoritative_sources"]`` overrides it.
    ontology_authoritative_sources: str = ""
    # "none" (default) uses the ontology's asserted triples only; "owl_rl" expands the
    # OWL 2 RL closure (inverse/transitive properties, domain/range typing,
    # equivalentClass, sameAs) over the parsed graph before the lookup is built.
    ontology_reasoning: str = DEFAULT_ONTOLOGY_REASONING

    model_config = SettingsConfigDict(extra="allow", populate_by_name=True)

    @field_validator("ontology_mode", mode="before")
    @classmethod
    def _normalize_ontology_mode(cls, value) -> str:
        return normalize_ontology_mode(value)

    def authoritative_sources(self) -> dict[str, str | None]:
        """Parse ``ontology_authoritative_sources`` into the per-call dict shape."""
        return parse_authoritative_sources(self.ontology_authoritative_sources)

    @field_validator("ontology_reasoning", mode="before")
    @classmethod
    def _normalize_ontology_reasoning(cls, value) -> str:
        return normalize_ontology_reasoning(value)

    def to_dict(self) -> dict:
        """
        Return the resolver-factory keyword arguments as a dictionary.

        Note: this dict is splatted into ``get_ontology_resolver_from_env`` — it must
        contain exactly that function's parameters. ``ontology_mode`` deliberately does
        not belong here; it is read separately via ``get_configured_ontology_mode``.
        ``ontology_reasoning`` does belong here: it decides how the resolver is built.
        """
        return {
            "ontology_resolver": self.ontology_resolver,
            "matching_strategy": self.matching_strategy,
            "ontology_file_path": self.ontology_file_path,
            "ontology_reasoning": self.ontology_reasoning,
        }


@lru_cache
def get_ontology_env_config():
    """
    Retrieve the ontology configuration. This function utilizes caching to return a
    singleton instance of the OntologyConfig class for efficiency.
    """
    return OntologyEnvConfig()

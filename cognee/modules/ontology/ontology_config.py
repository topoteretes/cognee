from typing import Any, TypedDict

from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.matching_strategies import MatchingStrategy


class OntologyConfig(TypedDict, total=False):
    """Configuration containing ontology resolver.

    Attributes:
        ontology_resolver: The ontology resolver instance to use
        ontology_mode: How strictly to apply the ontology for this call —
            "annotate" (enrich only, the default) or "strict" (drop extracted
            entities with no ontology grounding). Falls back to the
            ONTOLOGY_MODE environment value when omitted.
        authoritative_sources: Which schema tables are the system of record for
            the concept they realize, keyed by table name (bare or
            ``schema.table``). A value is ``True``, an owner string, or a dict
            ``{"authoritative": bool, "owner": str}``. Stamped on the
            ``realizes`` edge; retrieval prefers authoritative tables when
            several realize the same class. Falls back to
            ONTOLOGY_AUTHORITATIVE_SOURCES when omitted.
    """

    ontology_resolver: BaseOntologyResolver | None
    ontology_mode: str | None
    authoritative_sources: dict[str, Any] | None


class Config(TypedDict, total=False):
    """Top-level configuration dictionary.

    Attributes:
        ontology_config: Configuration containing ontology resolver
    """

    ontology_config: OntologyConfig | None

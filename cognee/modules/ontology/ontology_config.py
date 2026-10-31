from typing import TypedDict

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
        ontology_reasoning: Whether to expand the ontology's OWL 2 RL closure
            before matching — "none" (the default) or "owl_rl" (requires the
            optional ``owlrl`` package). Applies when cognee builds the resolver
            (from ``ontology_file_path`` or the environment); an explicit
            ``ontology_resolver`` instance keeps the reasoning it was built with.
            Falls back to the ONTOLOGY_REASONING environment value when omitted.
    """

    ontology_resolver: BaseOntologyResolver | None
    ontology_mode: str | None
    ontology_reasoning: str | None


class Config(TypedDict, total=False):
    """Top-level configuration dictionary.

    Attributes:
        ontology_config: Configuration containing ontology resolver
    """

    ontology_config: OntologyConfig | None

from typing import Optional

from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.matching_strategies import MatchingStrategy, FuzzyMatchingStrategy
from cognee.modules.ontology.ontology_config import OntologyConfig


def get_ontology_resolver(
    resolver: Optional[BaseOntologyResolver] = None,
    matching_strategy: Optional[MatchingStrategy] = None,
) -> OntologyConfig:
    """Get ontology resolver configuration with default or custom objects.

    Args:
        resolver: Optional pre-configured ontology resolver instance
        matching_strategy: Optional matching strategy instance

    Returns:
        Ontology configuration with default RDFLib resolver and fuzzy matching strategy,
        or custom objects if provided
    """
    config: OntologyConfig = {}

    if resolver is not None:
        config["resolver"] = resolver
    else:
        default_strategy = matching_strategy or FuzzyMatchingStrategy()
        config["resolver"] = RDFLibOntologyResolver(
            ontology_file=None, matching_strategy=default_strategy
        )

    if matching_strategy is not None and resolver is None:
        config["matching_strategy"] = matching_strategy

    return config

from typing import TypedDict, Optional

from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.matching_strategies import MatchingStrategy


class OntologyConfig(TypedDict, total=False):
    """Configuration for ontology resolver.

    Attributes:
        resolver: The ontology resolver instance to use
        matching_strategy: The matching strategy to use
    """

    resolver: Optional[BaseOntologyResolver]
    matching_strategy: Optional[MatchingStrategy]

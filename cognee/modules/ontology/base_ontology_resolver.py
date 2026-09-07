from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy, MatchingStrategy
from cognee.modules.ontology.models import AttachedOntologyNode


class BaseOntologyResolver(ABC):
    """Abstract base class for ontology resolvers."""

    def __init__(self, matching_strategy: Optional[MatchingStrategy] = None):
        """Initialize the ontology resolver with a matching strategy.

        Args:
            matching_strategy: The strategy to use for entity matching.
                              Defaults to FuzzyMatchingStrategy if None.
        """
        self.matching_strategy = matching_strategy or FuzzyMatchingStrategy()

    @abstractmethod
    def build_lookup(self) -> None:
        """Build the lookup dictionary for ontology entities."""

    @abstractmethod
    def refresh_lookup(self) -> None:
        """Refresh the lookup dictionary."""

    @abstractmethod
    def find_closest_match(self, name: str, category: str) -> Optional[str]:
        """Find the closest match for a given name in the specified category."""

    @abstractmethod
    def get_subgraph(
        self, node_name: str, node_type: str = "individuals", directed: bool = True
    ) -> Tuple[
        List[AttachedOntologyNode], List[Tuple[str, str, str]], Optional[AttachedOntologyNode]
    ]:
        """Get a subgraph for the given node."""

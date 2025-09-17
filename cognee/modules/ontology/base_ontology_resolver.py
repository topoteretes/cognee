from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from cognee.modules.ontology.models import AttachedOntologyNode


class BaseOntologyResolver(ABC):
    """Abstract base class for ontology resolvers."""

    @abstractmethod
    def build_lookup(self) -> None:
        """Build the lookup dictionary for ontology entities."""
        pass

    @abstractmethod
    def refresh_lookup(self) -> None:
        """Refresh the lookup dictionary."""
        pass

    @abstractmethod
    def find_closest_match(self, name: str, category: str) -> Optional[str]:
        """Find the closest match for a given name in the specified category."""
        pass

    @abstractmethod
    def get_subgraph(
        self, node_name: str, node_type: str = "individuals", directed: bool = True
    ) -> Tuple[List[AttachedOntologyNode], List[Tuple[str, str, str]], Optional[AttachedOntologyNode]]:
        """Get a subgraph for the given node."""
        pass

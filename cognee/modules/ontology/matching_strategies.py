import difflib
from abc import ABC, abstractmethod
from typing import List, Optional


class MatchingStrategy(ABC):
    """Abstract base class for ontology entity matching strategies."""

    @abstractmethod
    def find_match(self, name: str, candidates: List[str]) -> Optional[str]:
        """Find the best match for a given name from a list of candidates.

        Args:
            name: The name to match
            candidates: List of candidate names to match against

        Returns:
            The best matching candidate name, or None if no match found
        """
        pass


class FuzzyMatchingStrategy(MatchingStrategy):
    """Fuzzy matching strategy using difflib for approximate string matching."""

    def __init__(self, cutoff: float = 0.8):
        """Initialize fuzzy matching strategy.

        Args:
            cutoff: Minimum similarity score (0.0 to 1.0) for a match to be considered valid
        """
        self.cutoff = cutoff

    def find_match(self, name: str, candidates: List[str]) -> Optional[str]:
        """Find the closest fuzzy match for a given name.

        Args:
            name: The normalized name to match
            candidates: List of normalized candidate names

        Returns:
            The best matching candidate name, or None if no match meets the cutoff
        """
        if not candidates:
            return None

        # Check for exact match first
        if name in candidates:
            return name

        # Find fuzzy match
        best_match = difflib.get_close_matches(name, candidates, n=1, cutoff=self.cutoff)
        return best_match[0] if best_match else None

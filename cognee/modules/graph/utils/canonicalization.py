import json
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("canonicalization")


def strip_diacritics(s: str) -> str:
    """Remove diacritical marks (accents) from a string."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_alias_key(raw: str) -> str:
    """Normalize a raw string to alias-lookup form (Steps 1-3 only)."""
    s = unicodedata.normalize("NFKC", raw)
    s = strip_diacritics(s)
    s = s.lower()
    return s


@lru_cache(maxsize=None)
def load_alias_map(custom_path: Optional[str] = None) -> Dict[str, str]:
    """Load aliases from JSON. Keys must be in post-Step-3 form (see normalize_alias_key).

    Cached: the alias file is read once per path and the returned dict is treated
    read-only by callers.
    """
    default_path = os.path.join(os.path.dirname(__file__), "alias_maps", "default_aliases.json")
    path_to_load = custom_path or default_path

    try:
        with open(path_to_load, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("aliases", {})
    except FileNotFoundError:
        logger.warning(f"Alias map not found at {path_to_load}. Using empty map.")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse alias map at {path_to_load}: {e}")
        return {}


def canonicalize_entity_name(name: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    """
    Canonicalize an entity name into a deterministic grouping key.

    Pipeline:
      1. NFKC normalization (unicode decomposition/composition)
      2. Strip diacritics
      3. Lowercase
      4. Alias lookup (before punctuation is collapsed)
      5. Collapse whitespace, hyphens, dots, underscores to a single space
      6. Strip leading articles (the, a, an)
      7. Replace spaces with underscores

    Args:
        name: The raw entity name.
        alias_map: Optional dictionary of aliases. Keys must be normalized via normalize_alias_key.

    Returns:
        The canonicalized key. Two names that map to the same key are treated as
        the same entity by deterministic dedup.
    """
    if not name:
        return ""

    # Steps 1-3: NFKC + diacritics + lowercase
    s = unicodedata.normalize("NFKC", name)
    s = strip_diacritics(s)
    s = s.lower()

    # Step 4: Alias lookup (BEFORE punctuation collapse)
    # Note: alias_map keys must be stored in post-Step-3 form.
    # For example, "u.s.a." instead of "usa" or "u s a".
    if alias_map:
        s = alias_map.get(s, s)

    # Step 5: Collapse whitespace/hyphens/dots/underscores -> single space
    s = re.sub(r"[\s\-_.]+", " ", s).strip()

    # Step 6: Strip leading articles
    s = re.sub(r"^(the|a|an)\s+", "", s)

    # Step 7: Spaces -> underscore
    s = s.replace(" ", "_")

    return s

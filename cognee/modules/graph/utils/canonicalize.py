"""Versioned, deterministic identity canonicalization for rule-based dedup.

Approach A (deterministic, LLM-free). This is an *additive* layer: it does NOT
modify ``DataPoint._normalize_identity_value`` / ``id_for`` (doing so would
silently re-key every existing node). Instead it produces a richer canonical
form used to *block* and *merge* duplicate nodes inside
``deduplicate_nodes_and_edges`` — the surviving node keeps its original UUID5,
so existing graph references remain valid.

The canonicalizer is versioned so identities can be tagged with the version
that produced them and migrated non-destructively later.
"""

import re
import unicodedata

# Bump when the canonicalization rules change; stamp onto merges so a future
# migration can tell which nodes were keyed under which rule set.
CANON_VERSION = 1

# Configurable alias resolution. Keys are already-canonicalized surface forms
# (post NFKC / casefold / punctuation-strip / whitespace-collapse); values are
# the canonical form they resolve to. Kept in config-style data, not code
# branches, so it stays transparent and easy to extend.
ALIAS_MAP: dict[str, str] = {
    "nlp": "natural language processing",
    "n l p": "natural language processing",
    "natural language processing": "natural language processing",
    "ibm": "international business machines",
    "international business machines": "international business machines",
    "usa": "united states",
    "u s a": "united states",
    "united states of america": "united states",
    "united states": "united states",
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)  # hyphens, dots, etc. -> space
_WHITESPACE = re.compile(r"\s+")


def canonicalize(value: object, version: int = CANON_VERSION) -> str:
    """Return the deterministic canonical form of an identity value.

    Rules (all deterministic, no LLM):
      1. Unicode NFKC normalization (folds look-alikes / compatibility chars).
      2. Case folding.
      3. Punctuation (incl. hyphens) -> spaces.
      4. Whitespace collapsing + trim.
      5. Configurable alias resolution (abbreviations / known synonyms).
    """
    if not isinstance(value, str):
        return str(value)

    s = unicodedata.normalize("NFKC", value)
    s = s.casefold()
    s = _PUNCT.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return ALIAS_MAP.get(s, s)


def canonical_block_key(node: object) -> tuple:
    """Deterministic blocking key for a graph node.

    ``(ClassName, canonical(identity values))`` when the node declares
    ``identity_fields``; otherwise ``("__by_id__", str(id))`` so nodes without a
    stable identity fall back to the legacy id-equality behavior (never merged
    across ids). Two node types can never share a block.
    """
    metadata = getattr(node, "metadata", None) or {}
    identity_fields = metadata.get("identity_fields")
    if not identity_fields:
        return ("__by_id__", str(getattr(node, "id", id(node))))

    values = [canonicalize(getattr(node, f, None)) for f in identity_fields]
    return (type(node).__name__, "|".join(values))

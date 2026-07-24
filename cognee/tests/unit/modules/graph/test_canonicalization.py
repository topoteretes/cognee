"""Unit tests for entity-name canonicalization — issue #3627, Approach A."""

from cognee.modules.graph.utils.canonicalization import (
    canonicalize_entity_name,
    normalize_alias_key,
    load_alias_map,
)


def test_canonicalize_basic_normalization():
    # Dots/whitespace collapse to underscores; case and diacritics are folded.
    assert canonicalize_entity_name("U.S.A.") == "u_s_a"
    assert canonicalize_entity_name("Apple Inc.") == "apple_inc"
    assert canonicalize_entity_name("  whitespace  test  ") == "whitespace_test"
    assert canonicalize_entity_name("Café") == "cafe"
    assert canonicalize_entity_name("THE matrix") == "matrix"  # leading article stripped
    assert canonicalize_entity_name("") == ""


def test_canonicalize_with_alias_map():
    alias_map = {"u.s.a.": "united states", "usa": "united states"}
    assert canonicalize_entity_name("U.S.A.", alias_map) == "united_states"
    assert canonicalize_entity_name("USA", alias_map) == "united_states"
    assert canonicalize_entity_name("u.s.a.", alias_map) == "united_states"


def test_default_alias_map_unifies_country_variants():
    alias_map = load_alias_map()
    keys = {
        canonicalize_entity_name(name, alias_map)
        for name in ("USA", "U.S.A.", "United States", "United States of America")
    }
    assert keys == {"united_states"}


def test_normalize_alias_key_keeps_punctuation():
    # Alias keys are pre-collapse: only NFKC + diacritics + lowercase.
    assert normalize_alias_key("U.S.A.") == "u.s.a."
    assert normalize_alias_key("Café") == "cafe"

import pytest
from cognee.modules.graph.utils.canonicalization import canonicalize_entity_name, normalize_alias_key

def test_canonicalize_entity_name_basic_normalization():
    assert canonicalize_entity_name("U.S.A.") == "u.s.a."
    assert canonicalize_entity_name("U. S. A.") == "u_s_a_"  # Note: alias map is needed to catch this properly before collapse
    assert canonicalize_entity_name("Apple Inc.") == "apple_inc"
    assert canonicalize_entity_name("  whitespace  test  ") == "whitespace_test"
    assert canonicalize_entity_name("Café") == "cafe"
    assert canonicalize_entity_name("THE matrix") == "matrix"

def test_canonicalize_entity_name_with_alias_map():
    alias_map = {
        "u.s.a.": "united states",
        "usa": "united states"
    }
    
    assert canonicalize_entity_name("U.S.A.", alias_map) == "united_states"
    assert canonicalize_entity_name("USA", alias_map) == "united_states"
    assert canonicalize_entity_name("u.s.a.", alias_map) == "united_states"

def test_normalize_alias_key():
    assert normalize_alias_key("U.S.A.") == "u.s.a."
    assert normalize_alias_key("Café") == "cafe"

"""The string-level guards shared by query grounding, cognify and schema alignment."""

from types import SimpleNamespace

from cognee.modules.ontology.term_matching import (
    compact_key_index,
    find_class_named_in,
    head_phrase_tokens,
    lookup_words,
    multiword_match_is_sound,
    singular,
)


def test_multiword_match_rejects_a_swallowed_token():
    assert multiword_match_is_sound("enterprise_customer_acme", "EnterpriseCustomer") is False
    assert multiword_match_is_sound("enterprise customers", "EnterpriseCustomer") is True
    assert multiword_match_is_sound("credit exposure", "CreditExposure") is True
    # Single words keep the resolver's own judgement.
    assert multiword_match_is_sound("customers", "Customer") is True


def test_head_phrase_stops_at_the_first_preposition():
    assert head_phrase_tokens("credit exposure to acme corporation") == ["credit", "exposure"]
    assert head_phrase_tokens("customer feedback form") == ["customer", "feedback", "form"]
    assert head_phrase_tokens("of the customer") == ["the", "customer"]


def test_find_class_named_in_requires_the_class_to_be_the_head_noun():
    resolver = SimpleNamespace(
        lookup={"classes": dict.fromkeys(["customer", "creditexposure", "company", "form"])}
    )
    classes = compact_key_index(resolver, "classes")
    assert find_class_named_in("credit exposure to acme corporation", classes) == "creditexposure"
    assert find_class_named_in("credit exposures", classes) == "creditexposure"
    assert find_class_named_in("acme customer", classes) == "customer"
    # The head noun is "form", so this is not a Customer.
    assert find_class_named_in("customer feedback form", classes) == "form"
    assert find_class_named_in("enterprise customer acme", classes) is None
    assert find_class_named_in("", classes) is None


def test_find_class_named_in_skips_only_a_lone_stopword():
    resolver = SimpleNamespace(lookup={"classes": dict.fromkeys(["newcustomer", "will"])})
    classes = compact_key_index(resolver, "classes")
    assert find_class_named_in("acme new customer", classes) == "newcustomer"
    assert find_class_named_in("the last will", classes) is None


def test_lookup_words_tries_the_singular_and_skips_a_lone_stopword():
    resolver = SimpleNamespace(lookup={"classes": dict.fromkeys(["credit_exposure", "will"])})
    index = compact_key_index(resolver, "classes")
    assert lookup_words(["credit", "exposures"], index) == "credit_exposure"
    assert lookup_words(["will"], index) is None
    assert lookup_words([], index) is None


def test_singular_forms():
    assert singular("companies") == "company"
    assert singular("addresses") == "address"
    assert singular("customers") == "customer"
    assert singular("status") == "status"
    assert singular("bus") == "bus"

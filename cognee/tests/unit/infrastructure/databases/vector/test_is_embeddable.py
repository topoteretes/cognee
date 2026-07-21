from cognee.infrastructure.databases.vector.embeddings.utils import (
    is_embeddable,
    sanitize_embedding_text_inputs,
)


def test_non_whitespace_strings_are_embeddable():
    # string is embeddable when it has at least one non-whitespace character,
    # as documented in is_embeddable's docstring.
    assert is_embeddable("a") is True
    assert is_embeddable("  hi ") is True
    # punctuation/symbol-only strings have no alphanumeric character but are
    # still non-whitespace, so they are embeddable.
    assert is_embeddable("!!!") is True
    assert is_embeddable(".") is True


def test_empty_or_whitespace_only_strings_are_not_embeddable():
    assert is_embeddable("") is False
    assert is_embeddable("   ") is False


def test_non_string_inputs_are_not_embeddable():
    assert is_embeddable(None) is False
    assert is_embeddable(123) is False


def test_sanitize_replaces_only_empty_or_whitespace_inputs():
    # non-whitespace inputs are kept verbatim; empty/whitespace-only ones are
    # replaced with the "." dummy to avoid embedding-API errors.
    assert sanitize_embedding_text_inputs(["abc", "!!!", "", "   "]) == [
        "abc",
        "!!!",
        ".",
        ".",
    ]

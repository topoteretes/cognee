"""The pieces every lexical engine shares: tokenizer, passages, language setting (SDK-397)."""

import pytest

from cognee.infrastructure.databases.vector.lexical import (
    DEFAULT_STOP_WORDS,
    LEXICAL_SEARCH_LANGUAGES,
    split_passages,
    tokenize_words,
)


def test_tokenizer_lowercases_and_drops_stop_words():
    assert tokenize_words("The Rabbit, the HOLE!") == ["the", "rabbit", "the", "hole"]
    assert tokenize_words("The Rabbit, the HOLE!", DEFAULT_STOP_WORDS) == ["rabbit", "hole"]


def test_passages_pack_paragraphs_up_to_the_budget():
    text = "one two three\n\nfour five\n\nsix seven eight nine"
    # Paragraphs are packed while they fit the budget; a paragraph that would
    # overflow it starts the next passage. Without packing every paragraph
    # stands alone (snippets).
    assert split_passages(text, max_tokens=5) == [
        "one two three\n\nfour five",
        "six seven eight nine",
    ]
    assert split_passages(text, max_tokens=4) == [
        "one two three",
        "four five",
        "six seven eight nine",
    ]
    assert split_passages(text, max_tokens=5, pack=False) == [
        "one two three",
        "four five",
        "six seven eight nine",
    ]
    assert split_passages(text, max_tokens=100) == [text]


def test_an_oversized_paragraph_is_cut_at_sentences_then_words():
    paragraph = "Alpha beta gamma. Delta epsilon. " + "word " * 7
    passages = split_passages(paragraph, max_tokens=5)
    assert passages[0] == "Alpha beta gamma. Delta epsilon."
    assert all(len(passage.split()) <= 5 for passage in passages)
    assert " ".join(" ".join(passages).split()) == " ".join(paragraph.split()), "no words lost"


def test_blank_text_yields_no_passages_and_the_budget_is_validated():
    assert split_passages("  \n\n ") == []
    with pytest.raises(ValueError):
        split_passages("x", max_tokens=0)


def test_the_language_setting_is_validated():
    from cognee.infrastructure.databases.vector.config import VectorConfig

    assert LEXICAL_SEARCH_LANGUAGES == ("simple", "english")
    assert VectorConfig(_env_file=None).lexical_search_language == "simple"
    assert (
        VectorConfig(_env_file=None, lexical_search_language="English").lexical_search_language
        == "english"
    )
    with pytest.raises(ValueError, match="LEXICAL_SEARCH_LANGUAGE"):
        VectorConfig(_env_file=None, lexical_search_language="klingon")

import pytest

from cognee.infrastructure.llm.tokenizer.TikToken.adapter import TikTokenTokenizer


def test_decode_token_list_returns_text_for_each_token():
    """decode_token_list must decode each id, not pass a bare int to tiktoken.

    tiktoken's decode expects a sequence of ids, so decoding a list previously
    raised "'int' object is not an instance of 'Sequence'".
    """
    tokenizer = TikTokenTokenizer(model=None)

    tokens = tokenizer.extract_tokens("hello world foo")
    decoded = tokenizer.decode_token_list(tokens)

    assert all(isinstance(piece, str) for piece in decoded)
    assert "".join(decoded) == "hello world foo"


@pytest.mark.parametrize("model", [None, "gpt-4o"])
def test_special_token_spellings_are_ordinary_text(model):
    """Text spelling a special token (e.g. <|endoftext|>) is tokenized as ordinary text
    by every adapter method, instead of tiktoken raising on it (CLO-1384)."""
    tokenizer = TikTokenTokenizer(model=model, max_completion_tokens=5)
    encoding = tokenizer.tokenizer
    text = "\n".join(sorted(encoding.special_tokens_set))
    expected = encoding.encode_ordinary(text)

    assert tokenizer.extract_tokens(text) == expected
    assert tokenizer.count_tokens(text) == len(expected)
    assert tokenizer.trim_text_to_max_completion_tokens(text) == encoding.decode(expected[:5])


def test_a_space_that_ends_the_text_is_counted_where_running_text_has_it():
    """The chunker adds up the counts of words, each handed over with the space after it.

    That space belongs to whatever follows it. In front of a word it is part of the
    word's token (" was" is one token), in front of a number it is a token of its own
    (" 5" is two). Counted behind the word it was always a token of its own, so every
    word counted double.
    """
    tokenizer = TikTokenTokenizer(model="text-embedding-3-large")

    assert tokenizer.count_tokens("was") == 1
    assert tokenizer.count_tokens("was ") == 1
    assert tokenizer.count_tokens("5 ") == 2
    # White space alone is counted as it is, and so is any other white space.
    assert tokenizer.count_tokens(" ") == 1
    assert tokenizer.count_tokens("was\n") == 2
    assert tokenizer.count_tokens("") == 0


@pytest.mark.parametrize(
    "text",
    [
        "Alice was beginning to get very tired of sitting by her sister on the bank.",
        "12 345 6789 10 11 2026 7",
        "2026-11-23 17:05 INFO request 131 took 42 ms status 200",
    ],
    ids=["prose", "numbers", "log_line"],
)
def test_word_counts_add_up_to_the_text(text):
    """Never to less than the text holds: a chunk sized by the sum must fit its limit."""
    from cognee.tasks.chunks.chunk_by_word import chunk_by_word

    tokenizer = TikTokenTokenizer(model="text-embedding-3-large")
    whole = len(tokenizer.tokenizer.encode(text))

    added_up = sum(tokenizer.count_tokens(word) for word, _ in chunk_by_word(text))

    assert whole <= added_up <= whole + 1

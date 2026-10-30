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

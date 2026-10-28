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


@pytest.fixture(params=[None, "gpt-4o"])
def tokenizer(request):
    return TikTokenTokenizer(model=request.param)


@pytest.mark.parametrize("text", ["", "Ordinary document text.", "Literal <|endoftext|> content."])
def test_extract_and_count_tokens_preserve_text(tokenizer, text):
    expected = tokenizer.tokenizer.encode_ordinary(text)

    assert tokenizer.extract_tokens(text) == expected
    assert tokenizer.count_tokens(text) == len(expected)
    assert tokenizer.tokenizer.decode(tokenizer.extract_tokens(text)) == text


def test_special_token_spellings_are_ordinary_text(tokenizer):
    encoding = tokenizer.tokenizer
    text = "\n".join(sorted(encoding.special_tokens_set))
    special_ids = {
        token_id
        for marker in encoding.special_tokens_set
        for token_id in encoding.encode(marker, allowed_special={marker})
    }

    tokens = tokenizer.extract_tokens(text)

    assert tokens == encoding.encode_ordinary(text)
    assert not special_ids.intersection(tokens)
    assert encoding.decode(tokens) == text
    assert tokenizer.count_tokens(text) == len(tokens)


@pytest.mark.parametrize("limit", [0, 1, 10, 100])
def test_trim_literal_special_tokens_at_token_limit(tokenizer, limit):
    text = "Literal <|endoftext|> text followed by ordinary document content."
    tokens = tokenizer.tokenizer.encode_ordinary(text)
    tokenizer.max_completion_tokens = limit

    trimmed = tokenizer.trim_text_to_max_completion_tokens(text)

    assert trimmed == tokenizer.tokenizer.decode(tokens[:limit])
    assert tokenizer.count_tokens(trimmed) <= limit
    if len(tokens) <= limit:
        assert trimmed == text


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_trim_at_exact_token_boundary(tokenizer, offset):
    text = "Literal <|endoftext|> text."
    tokens = tokenizer.tokenizer.encode_ordinary(text)
    tokenizer.max_completion_tokens = len(tokens) + offset

    assert tokenizer.trim_text_to_max_completion_tokens(text) == tokenizer.tokenizer.decode(
        tokens[: tokenizer.max_completion_tokens]
    )

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

import pytest

from cognee.infrastructure.databases.vector.embeddings.utils import (
    _strip_surrogates,
    handle_embedding_response,
    is_embeddable,
    sanitize_embedding_text_inputs,
)


def test_is_embeddable_rejects_empty_and_whitespace():
    assert is_embeddable("") is False
    assert is_embeddable("   ") is False
    assert is_embeddable(123) is False


def test_is_embeddable_accepts_non_empty():
    assert is_embeddable("hello") is True
    assert is_embeddable("!") is True


def test_strip_surrogates_removes_unpaired_surrogate():
    poisoned = "some text \udc8f more text"
    cleaned = _strip_surrogates(poisoned)
    assert "\udc8f" not in cleaned
    cleaned.encode("utf-8")  # must not raise


def test_strip_surrogates_leaves_normal_text_unchanged():
    assert _strip_surrogates("hello world") == "hello world"
    assert _strip_surrogates("emoji 🎉 unicode café") == "emoji 🎉 unicode café"


def test_sanitize_embedding_text_inputs_strips_surrogates_in_valid_entries():
    result = sanitize_embedding_text_inputs(["ok \udc8f text", "", "   ", "fine"])
    assert "\udc8f" not in result[0]
    result[0].encode("utf-8")
    assert result[1] == "."
    assert result[2] == "."
    assert result[3] == "fine"


def test_sanitize_embedding_text_inputs_single_string():
    result = sanitize_embedding_text_inputs("just one \udc8f string")
    assert len(result) == 1
    assert "\udc8f" not in result[0]
    result[0].encode("utf-8")


def test_handle_embedding_response_zeroes_out_junk():
    original = ["", "valid"]
    embeddings = [[9.9, 9.9], [1.0, 2.0]]
    result = handle_embedding_response(original, embeddings, dimensions=2)
    assert result[0] == [0.0, 0.0]
    assert result[1] == [1.0, 2.0]

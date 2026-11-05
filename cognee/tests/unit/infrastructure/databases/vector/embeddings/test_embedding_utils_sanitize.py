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


def test_strip_surrogates_decodes_valid_utf16_surrogate_pair():
    pair = chr(0xD83D) + chr(0xDE00)
    assert _strip_surrogates(pair) == "\U0001f600"
    mixed = f"café {pair} done"
    assert _strip_surrogates(mixed) == "café \U0001f600 done"


def test_strip_surrogates_replaces_lone_high_and_low_surrogates():
    lone_high = f"before {chr(0xD83D)} after"
    lone_low = f"before {chr(0xDE00)} after"
    cleaned_high = _strip_surrogates(lone_high)
    cleaned_low = _strip_surrogates(lone_low)
    assert "\ufffd" in cleaned_high
    assert "\ufffd" in cleaned_low
    assert "\ud83d" not in cleaned_high
    assert "\ude00" not in cleaned_low
    cleaned_high.encode("utf-8")
    cleaned_low.encode("utf-8")


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

"""Chunkers are selected by a registered name; DEFAULT_CHUNKER names the default."""

import pytest

from cognee.exceptions import CogneeConfigurationError
from cognee.modules.chunking import chunker_registry
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.chunker_registry import (
    chunker_names,
    get_chunker,
    get_default_chunker,
    register_chunker,
    validate_chunker_name,
)
from cognee.modules.chunking.JsonListChunker import JsonListChunker
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import get_cognify_config


class ClauseChunker(Chunker):
    chunker_id = "clause_chunker_test"


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(chunker_registry, "registered_chunkers", {})
    monkeypatch.delenv("DEFAULT_CHUNKER", raising=False)
    get_cognify_config.cache_clear()
    yield
    get_cognify_config.cache_clear()


def test_built_in_chunkers_resolve_by_their_class_name():
    assert get_chunker("TextChunker") is TextChunker
    assert get_chunker("JsonListChunker") is JsonListChunker
    assert chunker_names() == ["CsvChunker", "JsonListChunker", "LangchainChunker", "TextChunker"]


def test_a_registered_chunker_resolves_and_is_listed():
    register_chunker("ClauseChunker", ClauseChunker)

    assert get_chunker("ClauseChunker") is ClauseChunker
    assert "ClauseChunker" in chunker_names()


def test_an_unknown_name_is_refused_and_names_what_is_available():
    with pytest.raises(
        ValueError, match="Unknown chunker 'cognee.evil.Chunker'. Available: .*TextChunker"
    ):
        get_chunker("cognee.evil.Chunker")


def test_a_name_is_validated_without_importing_the_chunker(monkeypatch):
    """LangchainChunker needs an optional dependency; naming it must not import it."""

    def refuse(module):
        raise AssertionError(f"imported {module}")

    monkeypatch.setattr(chunker_registry, "import_module", refuse)

    assert validate_chunker_name("LangchainChunker") == "LangchainChunker"
    with pytest.raises(ValueError, match="Unknown chunker"):
        validate_chunker_name("NoSuchChunker")


@pytest.mark.parametrize("name", ["", "   ", None])
def test_a_registered_name_must_be_a_non_empty_string(name):
    with pytest.raises(ValueError, match="non-empty string"):
        register_chunker(name, ClauseChunker)


def test_a_built_in_name_cannot_be_replaced():
    with pytest.raises(ValueError, match="built-in chunker"):
        register_chunker("TextChunker", ClauseChunker)

    assert get_chunker("TextChunker") is TextChunker


@pytest.mark.parametrize("not_a_chunker", [object, ClauseChunker(None, None, 1), "TextChunker"])
def test_only_a_chunker_class_can_be_registered(not_a_chunker):
    with pytest.raises(TypeError, match="subclass of Chunker"):
        register_chunker("Mine", not_a_chunker)


def test_the_default_is_text_chunker_unless_the_setting_names_another(monkeypatch):
    assert get_default_chunker() is TextChunker

    register_chunker("ClauseChunker", ClauseChunker)
    monkeypatch.setenv("DEFAULT_CHUNKER", "ClauseChunker")
    get_cognify_config.cache_clear()

    assert get_default_chunker() is ClauseChunker
    assert get_cognify_config().to_dict()["default_chunker"] == "ClauseChunker"


def test_a_default_that_is_not_registered_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("DEFAULT_CHUNKER", "ClauseChunker")
    get_cognify_config.cache_clear()

    with pytest.raises(
        CogneeConfigurationError, match="DEFAULT_CHUNKER is set to an unknown chunker"
    ):
        get_default_chunker()

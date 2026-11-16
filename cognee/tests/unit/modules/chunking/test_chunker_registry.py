"""DEFAULT_CHUNKER names the chunker a call uses when it passes none."""

import pytest

from cognee.exceptions import CogneeConfigurationError
from cognee.modules.chunking import chunker_registry
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.chunker_registry import get_default_chunker, register_chunker
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import get_cognify_config


class ClauseChunker(Chunker):
    chunker_id = "clause_chunker_test"


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(chunker_registry, "registered_chunkers", {"TextChunker": TextChunker})
    monkeypatch.delenv("DEFAULT_CHUNKER", raising=False)
    get_cognify_config.cache_clear()
    yield
    get_cognify_config.cache_clear()


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


def test_a_default_that_is_not_registered_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("DEFAULT_CHUNKER", "cognee.modules.chunking.CsvChunker")
    get_cognify_config.cache_clear()

    with pytest.raises(CogneeConfigurationError, match="DEFAULT_CHUNKER names an unknown chunker"):
        get_default_chunker()

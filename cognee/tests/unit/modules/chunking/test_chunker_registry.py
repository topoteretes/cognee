"""DEFAULT_CHUNKER names the chunker a call uses when it passes none."""

from types import SimpleNamespace

import pytest

from cognee.exceptions import CogneeConfigurationError
from cognee.modules.chunking import chunker_registry
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.chunker_registry import (
    get_default_chunker,
    register_chunker,
    resolve_chunker,
    stamp_chunk,
    stamps_every_chunk,
)
from cognee.modules.chunking.CsvChunker import CsvChunker
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


def test_a_chunker_without_a_chunker_id_cannot_be_registered():
    """update() tells chunkers apart by this id; without one it cannot refuse a mismatch."""

    class UnnamedChunker(Chunker):
        pass

    with pytest.raises(ValueError, match="UnnamedChunker declares no chunker_id"):
        register_chunker("Mine", UnnamedChunker)


def test_a_chunker_cannot_share_the_chunker_id_of_another_registered_one():
    """A subclass that changes the cut and keeps its parent's id would pass for the parent."""

    class TweakedTextChunker(TextChunker):
        pass

    with pytest.raises(ValueError, match="has the chunker_id of TextChunker"):
        register_chunker("Mine", TweakedTextChunker)


def test_a_class_can_be_registered_again_and_a_name_can_take_a_new_cut():
    class ClauseChunkerV2(Chunker):
        chunker_id = "clause_chunker_test_v2"

    register_chunker("ClauseChunker", ClauseChunker)
    register_chunker("ClauseChunker", ClauseChunker)
    register_chunker("ClauseChunker", ClauseChunkerV2)

    assert chunker_registry.registered_chunkers["ClauseChunker"] is ClauseChunkerV2


def test_a_name_cannot_take_another_class_with_the_same_chunker_id():
    """Documents the first class cut would pass for the second one's."""

    class ClauseChunkerRevised(Chunker):
        chunker_id = ClauseChunker.chunker_id

    register_chunker("ClauseChunker", ClauseChunker)
    with pytest.raises(ValueError, match="has the chunker_id of ClauseChunker"):
        register_chunker("ClauseChunker", ClauseChunkerRevised)


def test_the_name_text_chunker_cannot_be_repointed():
    class TweakedTextChunker(TextChunker):
        chunker_id = "tweaked_text_chunker_test"

    with pytest.raises(ValueError, match="names the built-in TextChunker"):
        register_chunker("TextChunker", TweakedTextChunker)
    register_chunker("TextChunker", TextChunker)


def test_a_chunker_cannot_take_the_id_of_a_built_in_chunker_that_is_not_registered():
    """Documents a per-call CsvChunker cut carry csv_chunker_v1 too."""

    class TweakedCsvChunker(CsvChunker):
        pass

    with pytest.raises(ValueError, match="has the chunker_id of CsvChunker"):
        register_chunker("Mine", TweakedCsvChunker)


def test_a_chunker_cannot_take_the_id_of_the_code_docs_pipeline():
    """The code-docs pipeline stamps its chunks with an id but has no chunker class."""

    class DocstringChunker(Chunker):
        chunker_id = "code_docs"

    with pytest.raises(ValueError, match="chunker_id of the code-docs pipeline"):
        register_chunker("Mine", DocstringChunker)


def test_a_call_that_names_no_chunker_gets_the_default(monkeypatch):
    assert resolve_chunker(None) is TextChunker
    assert resolve_chunker(ClauseChunker) is ClauseChunker

    register_chunker("ClauseChunker", ClauseChunker)
    monkeypatch.setenv("DEFAULT_CHUNKER", "ClauseChunker")
    get_cognify_config.cache_clear()
    assert resolve_chunker(None) is ClauseChunker


def test_a_blank_default_chunker_setting_counts_as_unset(monkeypatch):
    monkeypatch.setenv("DEFAULT_CHUNKER", "")
    get_cognify_config.cache_clear()

    assert get_default_chunker() is TextChunker


def test_a_chunk_cut_without_an_id_gets_its_chunkers_id():
    """A chunker that forgets to stamp cannot store a chunk no chunker owns."""
    for missing in (None, ""):
        chunk = SimpleNamespace(chunker_id=missing)
        stamp_chunk(ClauseChunker, chunk)
        assert chunk.chunker_id == ClauseChunker.chunker_id


def test_a_chunk_cut_with_another_chunkers_id_fails():
    """Stored with it, the document would be updated as if that chunker cut it."""
    stamp_chunk(ClauseChunker, SimpleNamespace(chunker_id=ClauseChunker.chunker_id))
    with pytest.raises(CogneeConfigurationError, match="ClauseChunker yielded a chunk"):
        stamp_chunk(ClauseChunker, SimpleNamespace(chunker_id=TextChunker.chunker_id))


def test_registration_does_not_change_how_a_chunker_stamps():
    """A per-call chunker follows the same rule as a registered one."""
    assert stamps_every_chunk(ClauseChunker)  # not registered: passed per call
    register_chunker("ClauseChunker", ClauseChunker)
    assert stamps_every_chunk(ClauseChunker)

    # TextChunker cut chunks before they carried an id, and keeps updating those.
    assert not stamps_every_chunk(TextChunker)


def test_a_chunker_without_an_id_leaves_its_chunks_without_one():
    class UnnamedChunker(Chunker):
        pass

    chunk = SimpleNamespace(chunker_id=None)
    stamp_chunk(UnnamedChunker, chunk)
    assert chunk.chunker_id is None
    assert not stamps_every_chunk(UnnamedChunker)


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

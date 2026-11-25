"""Journey: update() with the deployment's own default chunker (SDK-1525).

A deployment registers a chunker and names it in ``DEFAULT_CHUNKER``. Editing one
section of a document it cut must take the chunk-level path and extract that
section only. A document another chunker cut is rebuilt once, with the default,
and is chunk-level from its next update. That includes documents stored before
chunks carried a chunker id (cognee before 1.5.4), which TextChunker goes on
updating chunk-level. A chunker that leaves its id off its chunks gets them
stamped, so the rebuild does not fail after dropping the document's memory.
"""

from __future__ import annotations

import re

import pytest

import cognee
from cognee.api.v1.update.incremental import RefusalReason
from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking import chunker_registry
from cognee.modules.chunking.chunk_id import content_chunk_id
from cognee.modules.chunking.chunker_registry import register_chunker
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods import get_dataset_data
from cognee.tests.journeys import _support

DATASET = "journey_default_chunker_update"

WORKSHOPS = [
    ("Ropewalk", "Adaeze Wren"),
    ("Sailmaking", "Tomas Ferreira"),
    ("Cooperage", "Ingrid Solvay"),
    ("Chandlery", "Yusuf Haldane"),
    ("Smithy", "Olamide Farquhar"),
    ("Boatyard", "Mirela Okonkwo"),
]
DOC_V1 = "".join(
    f"# {name}\n\nThe {name.lower()} workshop in Quillon is run by {person}, who has kept its "
    f"ledgers since the harbour reopened and trains two apprentices every spring season.\n\n"
    for name, person in WORKSHOPS
)
DOC_V2 = DOC_V1.replace("Olamide Farquhar", "Bronagh Castellane")
DOC_V3 = DOC_V2.replace("Adaeze Wren", "Leocadia Brandt")


class SectionChunker(TextChunker):
    """One Markdown section per chunk; TextChunker cuts a section over the budget."""

    chunker_id = "section_chunker_v1"

    async def read(self):
        document_id = str(self.document.id)
        occurrences: dict[str, int] = {}
        async for text in self.get_text():
            for section in re.split(r"(?m)^(?=#)", text):
                if not section:
                    continue

                async def section_text(section=section):
                    yield section

                section_chunker = TextChunker(self.document, section_text, self.max_chunk_size)
                async for chunk in section_chunker.read():
                    occurrence = occurrences.get(chunk.content_hash, 0)
                    occurrences[chunk.content_hash] = occurrence + 1
                    chunk.id = content_chunk_id(document_id, chunk.content_hash, occurrence)
                    chunk.chunker_id = self.chunker_id
                    chunk.chunk_index = self.chunk_index
                    self.chunk_index += 1
                    yield chunk


class PreStampTextChunker(TextChunker):
    """TextChunker as it stored chunks before cognee 1.5.4: no chunker id on them."""

    # Chunkers had no id then, so nothing stamps one on these chunks.
    chunker_id = ""

    async def read(self):
        async for chunk in super().read():
            chunk.chunker_id = None
            yield chunk


class UnstampedSectionChunker(SectionChunker):
    """Declares an id, and leaves it off the chunks it yields."""

    chunker_id = "unstamped_section_chunker_v1"

    async def read(self):
        async for chunk in super().read():
            chunk.chunker_id = None
            yield chunk


@pytest.fixture
def use_chunker(monkeypatch):
    """Call the returned function to make a chunker the deployment's default."""
    monkeypatch.setattr(chunker_registry, "registered_chunkers", {"TextChunker": TextChunker})

    def switch(chunker=SectionChunker):
        register_chunker(chunker.__name__, chunker)
        monkeypatch.setenv("DEFAULT_CHUNKER", chunker.__name__)
        get_cognify_config.cache_clear()

    yield switch
    monkeypatch.delenv("DEFAULT_CHUNKER", raising=False)
    get_cognify_config.cache_clear()


async def _ingest(user, **cognify_arguments):
    await cognee.add(DOC_V1, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET], **cognify_arguments)
    dataset_id = await _support.dataset_id_by_name(DATASET, user)
    return dataset_id, (await get_dataset_data(dataset_id))[0]


async def _stored_chunks(dataset_id, user) -> list[dict]:
    """The document's chunk nodes, in document order."""
    async with set_database_global_context_variables(dataset_id, user.id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
    chunks = [properties for _, properties in nodes if properties.get("type") == "DocumentChunk"]
    return sorted(chunks, key=lambda chunk: chunk["chunk_index"])


def _graph_extractions(mock_llm) -> int:
    return sum(1 for model_name, _ in mock_llm.calls if model_name == "KnowledgeGraph")


@pytest.mark.journey
@pytest.mark.asyncio
async def test_update_replaces_only_the_edited_section(
    clean_env, default_user, mock_llm, use_chunker
):
    use_chunker()
    dataset_id, row = await _ingest(default_user)

    chunks = await _stored_chunks(dataset_id, default_user)
    assert [chunk["text"].splitlines()[0] for chunk in chunks] == [
        f"# {name}" for name, _ in WORKSHOPS
    ], "the default chunker did not cut the document at its headings"
    assert {chunk["chunker_id"] for chunk in chunks} == {SectionChunker.chunker_id}

    if mock_llm is not None:
        mock_llm.calls.clear()
    result = await cognee.update(
        data_id=row.id, data=DOC_V2, dataset_id=dataset_id, user=default_user
    )

    assert result["status"] == "incremental", f"update rebuilt the document: {result}"
    assert result["fallback"] is None
    assert (result["regions"], result["deleted_chunks"], result["added_chunks"]) == (1, 1, 1)
    assert result["kept_chunks"] == len(WORKSHOPS) - 1
    if mock_llm is not None:
        assert _graph_extractions(mock_llm) == 1, "more than the edited section was extracted"

    chunks = await _stored_chunks(dataset_id, default_user)
    assert "".join(chunk["text"] for chunk in chunks) == DOC_V2
    assert {chunk["chunker_id"] for chunk in chunks} == {SectionChunker.chunker_id}
    assert len(chunks) == len(WORKSHOPS)


@pytest.mark.journey
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_by, stored_id",
    [(TextChunker, TextChunker.chunker_id), (PreStampTextChunker, None)],
    ids=["text_chunker", "before_chunker_ids"],
)
async def test_a_changed_default_chunker_rebuilds_a_document_once(
    clean_env, default_user, mock_llm, use_chunker, stored_by, stored_id
):
    # Stored while TextChunker was the default, by this version or by one from
    # before chunks carried a chunker id. Several chunks, so an update that
    # patched the document instead of rebuilding it would leave both cuts in it.
    dataset_id, row = await _ingest(default_user, chunker=stored_by, chunk_size=48)
    chunks = await _stored_chunks(dataset_id, default_user)
    assert len(chunks) > len(WORKSHOPS)
    assert {chunk.get("chunker_id") for chunk in chunks} == {stored_id}

    use_chunker()

    rebuilt = await cognee.update(
        data_id=row.id, data=DOC_V2, dataset_id=dataset_id, user=default_user
    )
    assert rebuilt["status"] == "full_rebuild", rebuilt
    assert rebuilt["fallback"]["reason"] is RefusalReason.UNSUPPORTED_CHUNKER
    chunks = await _stored_chunks(dataset_id, default_user)
    assert {chunk["chunker_id"] for chunk in chunks} == {SectionChunker.chunker_id}
    assert "".join(chunk["text"] for chunk in chunks) == DOC_V2

    if mock_llm is not None:
        mock_llm.calls.clear()
    result = await cognee.update(
        data_id=row.id, data=DOC_V3, dataset_id=dataset_id, user=default_user
    )
    assert result["status"] == "incremental", f"the second update rebuilt again: {result}"
    if mock_llm is not None:
        assert _graph_extractions(mock_llm) == 1
    chunks = await _stored_chunks(dataset_id, default_user)
    assert "".join(chunk["text"] for chunk in chunks) == DOC_V3
    assert {chunk["chunker_id"] for chunk in chunks} == {SectionChunker.chunker_id}


@pytest.mark.journey
@pytest.mark.asyncio
async def test_chunks_stored_before_chunker_ids_stay_chunk_level_with_text_chunker(
    clean_env, default_user, mock_llm
):
    """A database from before cognee 1.5.4 keeps updating as it did."""
    dataset_id, row = await _ingest(default_user, chunker=PreStampTextChunker, chunk_size=48)
    stored = await _stored_chunks(dataset_id, default_user)
    assert {chunk.get("chunker_id") for chunk in stored} == {None}

    if mock_llm is not None:
        mock_llm.calls.clear()
    result = await cognee.update(
        data_id=row.id, data=DOC_V2, dataset_id=dataset_id, user=default_user
    )

    assert result["status"] == "incremental", f"update rebuilt the document: {result}"
    assert result["kept_chunks"] == len(stored) - 1
    if mock_llm is not None:
        assert _graph_extractions(mock_llm) == 1
    chunks = await _stored_chunks(dataset_id, default_user)
    assert "".join(chunk["text"] for chunk in chunks) == DOC_V2


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_chunker_that_does_not_stamp_its_chunks_gets_them_stamped(
    clean_env, default_user, mock_llm, use_chunker
):
    """The rebuild drops the document's memory before it cuts; the cut must not fail."""
    dataset_id, row = await _ingest(default_user, chunker=TextChunker, chunk_size=48)
    use_chunker(UnstampedSectionChunker)

    rebuilt = await cognee.update(
        data_id=row.id, data=DOC_V2, dataset_id=dataset_id, user=default_user
    )
    assert rebuilt["status"] == "full_rebuild", rebuilt
    chunks = await _stored_chunks(dataset_id, default_user)
    assert {chunk["chunker_id"] for chunk in chunks} == {UnstampedSectionChunker.chunker_id}

    result = await cognee.update(
        data_id=row.id, data=DOC_V3, dataset_id=dataset_id, user=default_user
    )
    assert result["status"] == "incremental", f"the stamped document was rebuilt: {result}"


@pytest.mark.journey
@pytest.mark.asyncio
async def test_remember_refuses_an_unregistered_default_chunker_before_storing(
    clean_env, default_user, mock_llm, use_chunker, monkeypatch
):
    """The setting names a chunker this process never registered: nothing is added."""
    monkeypatch.setenv("DEFAULT_CHUNKER", "NeverRegisteredChunker")
    get_cognify_config.cache_clear()

    with pytest.raises(CogneeConfigurationError, match="DEFAULT_CHUNKER names an unknown chunker"):
        await cognee.remember(DOC_V1, dataset_name=DATASET, self_improvement=False)

    dataset_id = await _support.dataset_id_by_name(DATASET, default_user)
    rows = await get_dataset_data(dataset_id) if dataset_id else []
    assert rows == [], "the document was added before the refusal"


DATED_V1 = (
    "# Founding\n\nThe Quillon ropewalk was founded on 3 March 1947 by Adaeze Wren.\n\n"
    "# Charter\n\nIts charter was signed by Tomas Ferreira in the harbour office.\n\n"
    "# Opening\n\nThe ropewalk opened to the public the following 27 April with a parade.\n\n"
)
DATED_V2 = DATED_V1.replace("1947", "1948")
DATED_V3 = DATED_V2.replace("Tomas Ferreira", "Ingrid Solvay")


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_chunk_re_dated_by_an_edit_keeps_its_chunker_id(
    clean_env, default_user, mock_llm, use_chunker
):
    """Changing a year re-dates a later section's "the following 27 April".

    The re-dated chunk is written again from the stored node; without its id the
    next update would find a chunk the default chunker did not stamp and rebuild.
    """
    use_chunker()
    await cognee.add(DATED_V1, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])
    dataset_id = await _support.dataset_id_by_name(DATASET, default_user)
    row = (await get_dataset_data(dataset_id))[0]

    redated = await cognee.update(
        data_id=row.id, data=DATED_V2, dataset_id=dataset_id, user=default_user
    )
    assert redated["status"] == "incremental", redated
    assert redated["redated_chunks"] == 1, "the edit did not re-date the later section"
    chunks = await _stored_chunks(dataset_id, default_user)
    assert {chunk["chunker_id"] for chunk in chunks} == {SectionChunker.chunker_id}

    following = await cognee.update(
        data_id=row.id, data=DATED_V3, dataset_id=dataset_id, user=default_user
    )
    assert following["status"] == "incremental", f"the next update rebuilt: {following}"

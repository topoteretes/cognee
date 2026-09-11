"""Dataset-scoped Data identity: per-dataset rows, lookup dedup, no bleed.

The refactor decouples document identity from content: ``Data.id`` is random,
rows are dataset-scoped (``Data.dataset_id``), and dedup is a lookup on
``(dataset_id, content_hash)``. This suite pins the new contract at the
ingestion layer (no cognify, no LLM):

  1. the same content in two datasets is two rows with two ids;
  2. THE BLEED REGRESSION: rewriting one dataset's document in place leaves
     the other dataset's row, stored text, and content hash untouched;
  3. re-adding content a dataset already holds reuses the row (idempotent);
  4. a data_id pinned from another dataset is refused outright — no linking,
     no mutation;
  5-9. every id ever issued keeps resolving: pre-fork ids resolve scoped
     and context-free (single fork), ambiguity refuses with candidates,
     pinned pre-fork ids bind to the canonical row, update() carries the
     original id onto its replacement row (flatten-on-write), and deletion
     by the pre-fork id works through the real API — after an update;
  10. dlt-derived ids are dataset-scoped: the same dlt row in two datasets is
     two id families, and pre-scoping rows are adopted in their own dataset
     only.

Runs on the default local stack (sqlite), no API keys.
"""

import asyncio
import re
import shutil
import tempfile
from pathlib import Path

import pytest

MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def scoped_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_scoped_data_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    def clear_config_caches():
        import importlib

        for module_name, factory_name in [
            ("cognee.base_config", "get_base_config"),
            ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
            (
                "cognee.infrastructure.databases.relational.get_relational_engine",
                "get_relational_engine",
            ),
            ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
            ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
            ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
            ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
            ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
            (
                "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
                "create_embedding_engine",
            ),
            ("cognee.infrastructure.llm.config", "get_llm_config"),
        ]:
            try:
                getattr(importlib.import_module(module_name), factory_name).cache_clear()
            except (ImportError, AttributeError):
                pass

    mp = pytest.MonkeyPatch()
    for key, value in dict(
        DB_PROVIDER="sqlite",
        VECTOR_DB_PROVIDER="lancedb",
        GRAPH_DATABASE_PROVIDER="kuzu",
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
    ).items():
        mp.setenv(key, value)
    clear_config_caches()

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(MARKER.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=n) for n in names], edges=[]
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    yield root

    LLMGateway.acreate_structured_output = original
    mp.undo()
    clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


def _para(tag: str) -> str:
    words = " ".join(f"{tag}{j:02d}" for j in range(12))
    return f"Paragraph {tag} ENT{tag.upper()} {words}.\n"


async def _dataset(name, user):
    from cognee.modules.data.methods import get_datasets

    return next(d for d in await get_datasets(user.id) if d.name == name)


async def _sole_data(dataset_id):
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data

    rows = await get_dataset_data(dataset_id)
    assert len(rows) == 1, f"expected one data row, got {len(rows)}"
    return rows[0]


async def _read_text(data_row) -> str:
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    async with open_data_file(data_row.raw_data_location) as file:
        content = file.read()
    return content.decode("utf-8") if isinstance(content, bytes) else content


def test_dataset_scoped_identity_and_bleed_regression(scoped_env):
    asyncio.run(_scenario())


async def _scenario():
    import cognee
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.ingestion.exceptions import IngestionError
    from cognee.modules.users.methods import get_default_user
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tasks.ingestion.ingest_data import ingest_data

    text_v1 = "".join(_para(tag) for tag in ["a", "b", "c", "d"])
    text_v2 = text_v1.replace("ENTC", "ENTC2", 1)

    # --- 1. same content, two datasets -> two rows, two ids ---------------- #
    await cognee.add(text_v1, dataset_name="alpha")
    await cognee.add(text_v1, dataset_name="beta")
    user = await get_default_user()
    alpha = await _dataset("alpha", user)
    beta = await _dataset("beta", user)

    alpha_data = await _sole_data(alpha.id)
    beta_data = await _sole_data(beta.id)
    assert alpha_data.id != beta_data.id, "same content in two datasets must be two identities"
    assert str(alpha_data.dataset_id) == str(alpha.id)
    assert str(beta_data.dataset_id) == str(beta.id)
    assert alpha_data.content_hash == beta_data.content_hash

    # --- 2. THE BLEED REGRESSION ------------------------------------------- #
    beta_text_before = await _read_text(beta_data)
    beta_hash_before = beta_data.content_hash

    await ingest_data(
        [DataItem(data=text_v2, data_id=alpha_data.id)], "alpha", user, None, alpha.id
    )

    alpha_after = await _sole_data(alpha.id)
    assert alpha_after.id == alpha_data.id, "in-place update must preserve the row id"
    assert await _read_text(alpha_after) == text_v2

    beta_after = await _sole_data(beta.id)
    assert beta_after.id == beta_data.id
    assert beta_after.content_hash == beta_hash_before, "beta's content hash must not move"
    assert await _read_text(beta_after) == beta_text_before, (
        "updating alpha's document must not rewrite beta's stored text"
    )

    # --- 3. idempotent re-add within a dataset ----------------------------- #
    await cognee.add(text_v2, dataset_name="alpha")
    alpha_rows = await get_dataset_data(alpha.id)
    assert len(alpha_rows) == 1 and alpha_rows[0].id == alpha_data.id, (
        "re-adding current content must reuse the row, not create a document"
    )

    # --- 3b. dedup is owner-scoped ----------------------------------------- #
    # In a shared multi-writer dataset, another user adding the same bytes must
    # NOT resolve to (and later overwrite ownership of) this user's row.
    from types import SimpleNamespace
    from uuid import uuid4

    import cognee.modules.ingestion as ingestion_module

    classified = ingestion_module.classify(text_v2)
    stranger = SimpleNamespace(id=uuid4(), tenant_id=None)
    assert await ingestion_module.identify(classified, stranger, alpha.id) is None, (
        "another owner's identical content must miss the dedup lookup"
    )
    assert await ingestion_module.identify(classified, user, alpha.id) == alpha_data.id, (
        "the owning user's identical content must hit their own row"
    )

    # --- 4. mispinned foreign row: refuse to touch ------------------------- #
    # A data_id pinned from another dataset must never be linked or mutated.
    with pytest.raises(IngestionError, match="is not available in dataset"):
        await ingest_data(
            [DataItem(data=text_v1, data_id=alpha_data.id)], "beta", user, None, beta.id
        )
    beta_rows = await get_dataset_data(beta.id)
    assert len(beta_rows) == 1 and beta_rows[0].id == beta_data.id
    assert await _read_text(beta_rows[0]) == beta_text_before, (
        "a refused foreign pin must leave the target dataset untouched"
    )

    # --- 5. every id ever issued keeps resolving --------------------------- #
    # Simulate a backfill-split fork: beta's row records a pre-fork original.
    from uuid import uuid4 as _mint
    from sqlalchemy import update as sql_update
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.exceptions import AmbiguousDataIdError
    from cognee.modules.data.methods import get_data, resolve_data_id
    from cognee.modules.data.models import Data

    pre_fork_id = _mint()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await session.execute(
            sql_update(Data).where(Data.id == beta_data.id).values(legacy_id=pre_fork_id)
        )
        await session.commit()

    assert await resolve_data_id(beta.id, beta_data.id) == beta_data.id, "exact id wins"
    assert await resolve_data_id(beta.id, pre_fork_id) == beta_data.id, (
        "the pre-fork id must resolve to the canonical row within its dataset"
    )
    assert await resolve_data_id(alpha.id, pre_fork_id) is None, (
        "the old id means nothing in an unrelated dataset"
    )

    # --- 6. get_data: context-free guard matrix ---------------------------- #
    row = await get_data(user.id, beta_data.id)
    assert row is not None and row.id == beta_data.id, "unforked exact id: unchanged behavior"

    row = await get_data(user.id, pre_fork_id)
    assert row is not None and row.id == beta_data.id, (
        "single fork, no exact match: unambiguous, returned directly"
    )

    row = await get_data(user.id, pre_fork_id, dataset_id=beta.id)
    assert row is not None and row.id == beta_data.id, "scoped resolution via legacy id"
    assert await get_data(user.id, pre_fork_id, dataset_id=alpha.id) is None

    # A second dataset forking from the SAME original -> ambiguity MUST refuse.
    async with engine.get_async_session() as session:
        await session.execute(
            sql_update(Data).where(Data.id == alpha_data.id).values(legacy_id=pre_fork_id)
        )
        await session.commit()
    try:
        await get_data(user.id, pre_fork_id)
        raise AssertionError("ambiguous context-free lookup must refuse, never guess")
    except AmbiguousDataIdError as error:
        candidate_ids = {str(c["data_id"]) for c in error.candidates}
        assert candidate_ids == {str(alpha_data.id), str(beta_data.id)}, error.candidates
    row = await get_data(user.id, pre_fork_id, dataset_id=alpha.id)
    assert row is not None and row.id == alpha_data.id, "dataset scope disambiguates"
    async with engine.get_async_session() as session:
        await session.execute(
            sql_update(Data).where(Data.id == alpha_data.id).values(legacy_id=None)
        )
        await session.commit()

    # --- 7. pinned DataItem with a pre-fork id ----------------------------- #
    beta_text_current = await _read_text(beta_data)
    await ingest_data(
        [DataItem(data=beta_text_current, data_id=pre_fork_id)], "beta", user, None, beta.id
    )
    beta_rows = await get_dataset_data(beta.id)
    assert len(beta_rows) == 1 and beta_rows[0].id == beta_data.id, (
        "a pinned pre-fork id must resolve to the canonical row, not mint one"
    )

    # --- 8. update() keeps the data_id: pinned re-ingest + lineage restore - #
    # A fork row updated by its PRE-FORK id: the row is recreated under the
    # SAME canonical id, with the fork lineage restored, so both ids the user
    # ever held keep resolving.
    text_v3 = text_v1.replace("ENTA", "ENTA3", 1)
    await cognee.update(pre_fork_id, text_v3, beta.id, user=user)
    beta_rows = await get_dataset_data(beta.id)
    assert len(beta_rows) == 1
    replacement = beta_rows[0]
    assert replacement.id == beta_data.id, "update() must NOT change the document's data_id"
    assert str(replacement.legacy_id) == str(pre_fork_id), (
        "the fork lineage is restored on the recreated row"
    )
    assert await _read_text(replacement) == text_v3
    assert await resolve_data_id(beta.id, pre_fork_id) == replacement.id, (
        "the pre-fork id survives the update"
    )
    row = await get_data(user.id, pre_fork_id)
    assert row is not None and row.id == replacement.id

    # A normal (unforked) row updated by its own id: same id, no lineage.
    text_v4 = text_v2.replace("ENTB", "ENTB4", 1)
    await cognee.update(alpha_data.id, text_v4, alpha.id, user=user)
    alpha_updated = await _sole_data(alpha.id)
    assert alpha_updated.id == alpha_data.id, "update() must NOT change the document's data_id"
    assert alpha_updated.legacy_id is None
    assert await _read_text(alpha_updated) == text_v4

    # Replacing one document with several items is ambiguous — refused.
    try:
        await cognee.update(alpha_data.id, ["one", "two"], alpha.id, user=user)
    except IngestionError:
        pass
    else:
        raise AssertionError("update() with a multi-item list must be refused")

    # --- 9. delete by the pre-fork id through the real API ----------------- #
    from cognee.api.v1.datasets.datasets import datasets as datasets_api

    await datasets_api.delete_data(beta.id, pre_fork_id, user=user)
    assert await get_dataset_data(beta.id) == [], "delete by pre-fork id must succeed"

    # --- 10. dlt-derived ids are dataset-scoped ----------------------------- #
    # The same dlt row loaded into two datasets must be two id families (a
    # shared id would trip ingestion's foreign-pin guard); rows ingested
    # before ids were dataset-namespaced are adopted in their own dataset only.
    from cognee.modules.data.methods.get_unique_data_id import get_unique_data_id
    from cognee.modules.data.models import Data
    from cognee.tasks.ingestion.dlt_row_data import DltRowData
    from cognee.tasks.ingestion.resolve_dlt_sources import _dlt_row_identifier, _stable_row_ids
    from sqlalchemy import insert as sql_insert

    dlt_row = DltRowData(
        table_name="users",
        primary_key_column="id",
        primary_key_value="1",
        row_data={"id": 1, "name": "Ada"},
        content_hash="hash-dlt-1",
        schema_info=[],
        schema_hash="schema-1",
        foreign_keys=[],
        dlt_db_name="dlt_db",
        dataset_name="alpha",
    )

    (alpha_dlt_id,) = await _stable_row_ids([dlt_row], user, alpha.id)
    (beta_dlt_id,) = await _stable_row_ids([dlt_row], user, beta.id)
    assert alpha_dlt_id != beta_dlt_id, "same dlt row in two datasets must be two ids"
    assert (await _stable_row_ids([dlt_row], user, alpha.id)) == [alpha_dlt_id], (
        "dlt ids are stable across derivations"
    )

    # A pre-scoping row (old derivation as its primary id) is adopted in its
    # own dataset — and never from another dataset.
    pre_scoping_id = await get_unique_data_id(_dlt_row_identifier(dlt_row), user)
    from cognee.infrastructure.databases.relational import get_relational_engine

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await session.execute(
            sql_insert(Data.__table__).values(
                id=pre_scoping_id,
                dataset_id=alpha.id,
                name="dlt-legacy-row",
                content_hash="hash-dlt-1",
                raw_data_location="file:///tmp/dlt-legacy-row.txt",
                owner_id=user.id,
                pipeline_status={},
                token_count=-1,
            )
        )
        await session.commit()
    try:
        assert (await _stable_row_ids([dlt_row], user, alpha.id)) == [pre_scoping_id], (
            "a pre-scoping dlt row keeps its id in its own dataset"
        )
        assert (await _stable_row_ids([dlt_row], user, beta.id)) == [beta_dlt_id], (
            "another dataset never adopts a foreign pre-scoping row"
        )
    finally:
        async with engine.get_async_session() as session:
            await session.execute(Data.__table__.delete().where(Data.id == pre_scoping_id))
            await session.commit()

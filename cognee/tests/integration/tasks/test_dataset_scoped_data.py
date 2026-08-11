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
  5. every id ever issued keeps resolving: a backfill-split row is found
     (and deletable through the real API) by its pre-split id. (Legacy-row semantics live in the backfill migration test:
     tests/unit/modules/data/test_dataset_scoped_backfill.py.)

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
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_scoped_data_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        DB_PROVIDER="sqlite",
        VECTOR_DB_PROVIDER="lancedb",
        GRAPH_DATABASE_PROVIDER="kuzu",
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
    )

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

    yield root

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
    with pytest.raises(IngestionError, match="belongs to dataset"):
        await ingest_data(
            [DataItem(data=text_v1, data_id=alpha_data.id)], "beta", user, None, beta.id
        )
    beta_rows = await get_dataset_data(beta.id)
    assert len(beta_rows) == 1 and beta_rows[0].id == beta_data.id
    assert await _read_text(beta_rows[0]) == beta_text_before, (
        "a refused foreign pin must leave the target dataset untouched"
    )

    # --- 5. every id ever issued keeps resolving --------------------------- #
    # Simulate a backfill-split row: beta's row records a pre-split original
    # id. Resolution must find the row by the OLD id within the dataset.
    from uuid import uuid4 as _mint
    from sqlalchemy import update as sql_update
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.methods import resolve_data_id
    from cognee.modules.data.models import Data

    pre_split_id = _mint()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await session.execute(
            sql_update(Data).where(Data.id == beta_data.id).values(split_from_data_id=pre_split_id)
        )
        await session.commit()

    assert await resolve_data_id(beta.id, beta_data.id) == beta_data.id, "exact id wins"
    assert await resolve_data_id(beta.id, pre_split_id) == beta_data.id, (
        "the pre-split id must resolve to the canonical row within its dataset"
    )
    assert await resolve_data_id(alpha.id, pre_split_id) is None, (
        "the old id means nothing in an unrelated dataset"
    )

    # And the real entry point honors it: delete by the OLD id removes the row.
    from cognee.api.v1.datasets.datasets import datasets as datasets_api

    await datasets_api.delete_data(beta.id, pre_split_id, user=user)
    assert await get_dataset_data(beta.id) == [], "delete by pre-split id must succeed"

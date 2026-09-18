"""update()'s full rebuild is a PATCH for document metadata (SDK-750).

A rebuild re-adds the document pinned to its id. Before this fix the re-add rebuilt
the row from the request alone, so a label, external_metadata or node_set the caller
did not resend was wiped. Now: absent keeps, an explicit empty value clears, a value
replaces. Runs on the mocked-LLM incremental fixture, no keys needed.
"""

import asyncio
import contextvars

import pytest

from cognee.tests.e2e.incremental_update.backend_env import reset_backend_state
from cognee.tests.e2e.incremental_update.test_incremental_update import (
    CHUNK_TOKENS,
    _paragraph,
)
from cognee.tests.e2e.incremental_update.test_incremental_update import (
    incremental_env as _incremental_env,
)

# Reuse the module-scoped fixture (scratch roots, env, LLM mock) under its own name.
incremental_env = _incremental_env


async def _row(data_id):
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    async with get_relational_engine().get_async_session() as session:
        row = await session.get(Data, data_id)
        return row.label, row.external_metadata, row.node_set


@pytest.mark.asyncio
async def test_full_rebuild_keeps_clears_and_replaces_document_metadata(incremental_env):
    await reset_backend_state()
    import cognee
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user
    from cognee.tasks.ingestion.data_item import DataItem

    pristine = contextvars.copy_context()

    async def update(*args, **kwargs):
        # Like an API request: a fresh context per call, so dataset context vars
        # set by one update never leak into the next.
        loop = asyncio.get_running_loop()
        return await loop.create_task(cognee.update(*args, **kwargs), context=pristine.copy())

    text = "".join(_paragraph(i) for i in range(4))
    await cognee.add(
        DataItem(data=text, label="doc-label", external_metadata={"source": "audit"}),
        dataset_name="meta_patch",
        node_set=["ns1"],
    )
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "meta_patch")
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)
    data_id = (await get_dataset_data(dataset.id))[0].id

    assert await _row(data_id) == ("doc-label", {"source": "audit", "node_set": ["ns1"]}, '["ns1"]')

    # Keep: a forced full rebuild that sends nothing about metadata changes nothing.
    result = await update(data_id, text, dataset.id, user, chunk_level_diff=False)
    assert result["status"] == "full_rebuild"
    assert await _row(data_id) == ("doc-label", {"source": "audit", "node_set": ["ns1"]}, '["ns1"]')

    # Keep: resending only the node set keeps label and metadata.
    await update(data_id, text, dataset.id, user, node_set=["ns1"])
    assert await _row(data_id) == ("doc-label", {"source": "audit", "node_set": ["ns1"]}, '["ns1"]')

    # Clear metadata: an explicit empty dict; label and node set stay.
    await update(data_id, DataItem(data=text, external_metadata={}), dataset.id, user)
    assert await _row(data_id) == ("doc-label", {"node_set": ["ns1"]}, '["ns1"]')

    # Clear node set: an explicit empty list; label stays.
    await update(data_id, text, dataset.id, user, node_set=[])
    assert await _row(data_id) == ("doc-label", {}, None)

    # Replace: values overwrite.
    await update(
        data_id,
        DataItem(data=text, label="new-label", external_metadata={"source": "crm"}),
        dataset.id,
        user,
        node_set=["ns2"],
    )
    assert await _row(data_id) == ("new-label", {"source": "crm", "node_set": ["ns2"]}, '["ns2"]')

    # Clear label: an explicit empty string.
    await update(data_id, DataItem(data=text, label=""), dataset.id, user)
    assert await _row(data_id) == ("", {"source": "crm", "node_set": ["ns2"]}, '["ns2"]')

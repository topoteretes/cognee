"""The background path copies a DataItem whole, including fields added later."""

from dataclasses import fields
from io import BytesIO
from uuid import uuid4

import pytest

from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.utils import materialize_stream_for_background


@pytest.mark.asyncio
async def test_every_data_item_field_survives_materialization():
    original = DataItem(
        data=BytesIO(b"payload"),
        label="doc",
        external_metadata={"k": "v"},
        system_metadata={"source": "notion"},
        data_id=uuid4(),
        node_set=["notion:ws:root"],
    )

    materialized = await materialize_stream_for_background(original)

    stream = getattr(materialized.data, "file", materialized.data)
    assert stream.read() == b"payload"
    for field in fields(DataItem):
        if field.name != "data":
            assert getattr(materialized, field.name) == getattr(original, field.name), field.name


@pytest.mark.asyncio
async def test_node_set_defaults_to_none_when_absent():
    materialized = await materialize_stream_for_background(DataItem(data="plain text"))
    assert materialized.node_set is None

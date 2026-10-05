"""materialize_stream_for_background must copy every DataItem field.

It uses dataclasses.replace so a new field can never be silently dropped the
way a hand-written field-by-field copy could.
"""

from uuid import uuid4

import pytest

from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.utils import materialize_stream_for_background


@pytest.mark.asyncio
async def test_every_data_item_field_survives_materialization():
    data_id = uuid4()
    original = DataItem(
        data="plain text",
        label="my-label",
        external_metadata={"k": "v"},
        system_metadata={"source": "notion"},
        data_id=data_id,
        literal_text=True,
    )

    materialized = await materialize_stream_for_background(original)

    assert materialized.data == "plain text"
    assert materialized.label == "my-label"
    assert materialized.external_metadata == {"k": "v"}
    assert materialized.system_metadata == {"source": "notion"}
    assert materialized.data_id == data_id
    assert materialized.literal_text is True


@pytest.mark.asyncio
async def test_literal_text_defaults_to_false_when_absent():
    original = DataItem(data="plain text")
    materialized = await materialize_stream_for_background(original)
    assert materialized.literal_text is False

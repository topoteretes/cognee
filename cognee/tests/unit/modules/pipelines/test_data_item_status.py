"""is_data_item_completed must accept both per-item status representations (SDK-23).

The per-item slot ``Data.pipeline_status[pipeline][str(dataset_id)]`` is today
the bare string ``DATA_ITEM_PROCESSING_COMPLETED``; the hash-aware rollout plans a
dict carrying that string under ``"status"``. Every reader that counts or skips
completed items should go through this one predicate so the two shapes cannot
drift apart.
"""

import pytest

from cognee.modules.pipelines.models import DataItemStatus, is_data_item_completed

COMPLETED = "DATA_ITEM_PROCESSING_COMPLETED"


@pytest.mark.parametrize(
    "value",
    [
        DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED,
        COMPLETED,
        {"status": COMPLETED},
        {"status": DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED},
        {"status": COMPLETED, "content_hash": "abc123", "token_count": 42},
    ],
)
def test_completed_representations(value):
    assert is_data_item_completed(value) is True


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "DATA_ITEM_PROCESSING_STARTED",
        "completed",
        {},
        {"status": None},
        {"status": "DATA_ITEM_PROCESSING_STARTED"},
        {"content_hash": "abc123"},
        ["DATA_ITEM_PROCESSING_COMPLETED"],
        0,
    ],
)
def test_not_completed_representations(value):
    assert is_data_item_completed(value) is False


def test_is_item_stamped_reads_the_pipeline_and_dataset_slot():
    from types import SimpleNamespace
    from uuid import uuid4

    from cognee.modules.pipelines.models.DataItemStatus import is_item_stamped

    dataset_id, other = uuid4(), uuid4()
    item = SimpleNamespace(
        pipeline_status={
            "indexing_pipeline": {str(dataset_id): {"status": COMPLETED}},
            "cognify_pipeline": {str(other): COMPLETED},
        }
    )
    assert is_item_stamped(item, "indexing_pipeline", dataset_id)
    assert not is_item_stamped(item, "cognify_pipeline", dataset_id)
    assert not is_item_stamped(item, "add_pipeline", dataset_id)
    assert not is_item_stamped(SimpleNamespace(pipeline_status=None), "indexing_pipeline", other)
    assert not is_item_stamped(object(), "indexing_pipeline", other)

from cognee.modules.pipelines.models.DataItem import DataItem
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

def test_data_item_label_field():
    item = DataItem(
        id="123",
        name="Sample Item",
        source="mock_source",
        status=DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED,
        label="Important"
    )
    assert item.label == "Important"

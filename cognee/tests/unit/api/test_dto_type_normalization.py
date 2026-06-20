from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from cognee.api.dto_types import (
    _coerce_optional_str_list,
    _coerce_optional_uuid_list,
)
from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO
from cognee.api.v1.recall.routers.get_recall_router import RecallPayloadDTO
from cognee.api.v1.search.routers.get_search_router import SearchPayloadDTO


class TestCoercionHelpers:
    def test_coerce_optional_str_list_single_string(self):
        assert _coerce_optional_str_list("alpha") == ["alpha"]

    def test_coerce_optional_str_list_list(self):
        assert _coerce_optional_str_list(["a", "b"]) == ["a", "b"]

    def test_coerce_optional_str_list_none_and_empty(self):
        assert _coerce_optional_str_list(None) is None
        assert _coerce_optional_str_list("") is None
        assert _coerce_optional_str_list("   ") is None

    def test_coerce_optional_str_list_rejects_invalid_type(self):
        with pytest.raises(ValueError):
            _coerce_optional_str_list(42)

    def test_coerce_optional_uuid_list_single_uuid(self):
        value = uuid4()
        assert _coerce_optional_uuid_list(value) == [value]

    def test_coerce_optional_uuid_list_string_uuid(self):
        value = uuid4()
        assert _coerce_optional_uuid_list(str(value)) == [value]

    def test_coerce_optional_uuid_list_list(self):
        first = uuid4()
        second = uuid4()
        assert _coerce_optional_uuid_list([first, second]) == [first, second]


class TestCognifyPayloadDTO:
    def test_accepts_single_dataset_name(self):
        payload = CognifyPayloadDTO(datasets="shared_dataset")
        assert payload.datasets == ["shared_dataset"]
        assert payload.dataset_ids is None

    def test_accepts_single_dataset_id(self):
        dataset_id = uuid4()
        payload = CognifyPayloadDTO(dataset_ids=dataset_id)
        assert payload.dataset_ids == [dataset_id]

    def test_accepts_dataset_name_list(self):
        payload = CognifyPayloadDTO(datasets=["a", "b"])
        assert payload.datasets == ["a", "b"]


class TestSearchPayloadDTO:
    def test_accepts_single_dataset_and_node_name(self):
        payload = SearchPayloadDTO(
            query="q",
            datasets="main_dataset",
            node_name="tech_docs",
        )
        assert payload.datasets == ["main_dataset"]
        assert payload.node_name == ["tech_docs"]

    def test_accepts_single_dataset_id(self):
        dataset_id = uuid4()
        payload = SearchPayloadDTO(query="q", dataset_ids=dataset_id)
        assert payload.dataset_ids == [dataset_id]


class TestRecallPayloadDTO:
    def test_accepts_single_values(self):
        dataset_id = uuid4()
        payload = RecallPayloadDTO(
            query="q",
            datasets="main_dataset",
            dataset_ids=dataset_id,
            node_name="memories",
        )
        assert payload.datasets == ["main_dataset"]
        assert payload.dataset_ids == [dataset_id]
        assert payload.node_name == ["memories"]

    def test_rejects_invalid_dataset_id_type(self):
        with pytest.raises(ValidationError):
            RecallPayloadDTO(query="q", dataset_ids=12345)

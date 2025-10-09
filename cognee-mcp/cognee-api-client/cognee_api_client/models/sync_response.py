from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="SyncResponse")


@_attrs_define
class SyncResponse:
    """Response model for sync operations.

    Attributes:
        run_id (str):
        status (str):
        dataset_ids (list[str]):
        dataset_names (list[str]):
        message (str):
        timestamp (str):
        user_id (str):
    """

    run_id: str
    status: str
    dataset_ids: list[str]
    dataset_names: list[str]
    message: str
    timestamp: str
    user_id: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        run_id = self.run_id

        status = self.status

        dataset_ids = self.dataset_ids

        dataset_names = self.dataset_names

        message = self.message

        timestamp = self.timestamp

        user_id = self.user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "run_id": run_id,
                "status": status,
                "dataset_ids": dataset_ids,
                "dataset_names": dataset_names,
                "message": message,
                "timestamp": timestamp,
                "user_id": user_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        run_id = d.pop("run_id")

        status = d.pop("status")

        dataset_ids = cast(list[str], d.pop("dataset_ids"))

        dataset_names = cast(list[str], d.pop("dataset_names"))

        message = d.pop("message")

        timestamp = d.pop("timestamp")

        user_id = d.pop("user_id")

        sync_response = cls(
            run_id=run_id,
            status=status,
            dataset_ids=dataset_ids,
            dataset_names=dataset_names,
            message=message,
            timestamp=timestamp,
            user_id=user_id,
        )

        sync_response.additional_properties = d
        return sync_response

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

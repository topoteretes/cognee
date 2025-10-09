from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SyncRequest")


@_attrs_define
class SyncRequest:
    """Request model for sync operations.

    Attributes:
        dataset_ids (Union[None, Unset, list[UUID]]):
    """

    dataset_ids: Union[None, Unset, list[UUID]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset_ids: Union[None, Unset, list[str]]
        if isinstance(self.dataset_ids, Unset):
            dataset_ids = UNSET
        elif isinstance(self.dataset_ids, list):
            dataset_ids = []
            for dataset_ids_type_0_item_data in self.dataset_ids:
                dataset_ids_type_0_item = str(dataset_ids_type_0_item_data)
                dataset_ids.append(dataset_ids_type_0_item)

        else:
            dataset_ids = self.dataset_ids

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if dataset_ids is not UNSET:
            field_dict["datasetIds"] = dataset_ids

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_dataset_ids(data: object) -> Union[None, Unset, list[UUID]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                dataset_ids_type_0 = []
                _dataset_ids_type_0 = data
                for dataset_ids_type_0_item_data in _dataset_ids_type_0:
                    dataset_ids_type_0_item = UUID(dataset_ids_type_0_item_data)

                    dataset_ids_type_0.append(dataset_ids_type_0_item)

                return dataset_ids_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[UUID]], data)

        dataset_ids = _parse_dataset_ids(d.pop("datasetIds", UNSET))

        sync_request = cls(
            dataset_ids=dataset_ids,
        )

        sync_request.additional_properties = d
        return sync_request

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

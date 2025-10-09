from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="SearchResult")


@_attrs_define
class SearchResult:
    """
    Attributes:
        search_result (Any):
        dataset_id (Union[None, UUID]):
        dataset_name (Union[None, str]):
    """

    search_result: Any
    dataset_id: Union[None, UUID]
    dataset_name: Union[None, str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        search_result = self.search_result

        dataset_id: Union[None, str]
        if isinstance(self.dataset_id, UUID):
            dataset_id = str(self.dataset_id)
        else:
            dataset_id = self.dataset_id

        dataset_name: Union[None, str]
        dataset_name = self.dataset_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "search_result": search_result,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        search_result = d.pop("search_result")

        def _parse_dataset_id(data: object) -> Union[None, UUID]:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                dataset_id_type_0 = UUID(data)

                return dataset_id_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, UUID], data)

        dataset_id = _parse_dataset_id(d.pop("dataset_id"))

        def _parse_dataset_name(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        dataset_name = _parse_dataset_name(d.pop("dataset_name"))

        search_result = cls(
            search_result=search_result,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
        )

        search_result.additional_properties = d
        return search_result

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

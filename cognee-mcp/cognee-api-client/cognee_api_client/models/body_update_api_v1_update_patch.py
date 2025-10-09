from collections.abc import Mapping
from io import BytesIO
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from .. import types
from ..types import UNSET, File, FileTypes, Unset

T = TypeVar("T", bound="BodyUpdateApiV1UpdatePatch")


@_attrs_define
class BodyUpdateApiV1UpdatePatch:
    """
    Attributes:
        data (Union[Unset, list[File]]):
        node_set (Union[None, Unset, list[str]]):
    """

    data: Union[Unset, list[File]] = UNSET
    node_set: Union[None, Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: Union[Unset, list[FileTypes]] = UNSET
        if not isinstance(self.data, Unset):
            data = []
            for data_item_data in self.data:
                data_item = data_item_data.to_tuple()

                data.append(data_item)

        node_set: Union[None, Unset, list[str]]
        if isinstance(self.node_set, Unset):
            node_set = UNSET
        elif isinstance(self.node_set, list):
            node_set = self.node_set

        else:
            node_set = self.node_set

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if data is not UNSET:
            field_dict["data"] = data
        if node_set is not UNSET:
            field_dict["node_set"] = node_set

        return field_dict

    def to_multipart(self) -> types.RequestFiles:
        files: types.RequestFiles = []

        if not isinstance(self.data, Unset):
            for data_item_element in self.data:
                files.append(("data", data_item_element.to_tuple()))

        if not isinstance(self.node_set, Unset):
            if isinstance(self.node_set, list):
                for node_set_type_0_item_element in self.node_set:
                    files.append(("node_set", (None, str(node_set_type_0_item_element).encode(), "text/plain")))
            else:
                files.append(("node_set", (None, str(self.node_set).encode(), "text/plain")))

        for prop_name, prop in self.additional_properties.items():
            files.append((prop_name, (None, str(prop).encode(), "text/plain")))

        return files

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        data = []
        _data = d.pop("data", UNSET)
        for data_item_data in _data or []:
            data_item = File(payload=BytesIO(data_item_data))

            data.append(data_item)

        def _parse_node_set(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                node_set_type_0 = cast(list[str], data)

                return node_set_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        node_set = _parse_node_set(d.pop("node_set", UNSET))

        body_update_api_v1_update_patch = cls(
            data=data,
            node_set=node_set,
        )

        body_update_api_v1_update_patch.additional_properties = d
        return body_update_api_v1_update_patch

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

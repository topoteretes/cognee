from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.tool_call_output_data_type_0 import ToolCallOutputDataType0


T = TypeVar("T", bound="ToolCallOutput")


@_attrs_define
class ToolCallOutput:
    """Output of a tool call in the responses API

    Attributes:
        status (Union[Unset, str]):  Default: 'success'.
        data (Union['ToolCallOutputDataType0', None, Unset]):
    """

    status: Union[Unset, str] = "success"
    data: Union["ToolCallOutputDataType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.tool_call_output_data_type_0 import ToolCallOutputDataType0

        status = self.status

        data: Union[None, Unset, dict[str, Any]]
        if isinstance(self.data, Unset):
            data = UNSET
        elif isinstance(self.data, ToolCallOutputDataType0):
            data = self.data.to_dict()
        else:
            data = self.data

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if status is not UNSET:
            field_dict["status"] = status
        if data is not UNSET:
            field_dict["data"] = data

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.tool_call_output_data_type_0 import ToolCallOutputDataType0

        d = dict(src_dict)
        status = d.pop("status", UNSET)

        def _parse_data(data: object) -> Union["ToolCallOutputDataType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                data_type_0 = ToolCallOutputDataType0.from_dict(data)

                return data_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ToolCallOutputDataType0", None, Unset], data)

        data = _parse_data(d.pop("data", UNSET))

        tool_call_output = cls(
            status=status,
            data=data,
        )

        tool_call_output.additional_properties = d
        return tool_call_output

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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.function import Function


T = TypeVar("T", bound="ToolFunction")


@_attrs_define
class ToolFunction:
    """Tool function wrapper (for OpenAI compatibility)

    Attributes:
        function (Function): Function definition compatible with OpenAI's format
        type_ (Union[Unset, str]):  Default: 'function'.
    """

    function: "Function"
    type_: Union[Unset, str] = "function"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        function = self.function.to_dict()

        type_ = self.type_

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "function": function,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.function import Function

        d = dict(src_dict)
        function = Function.from_dict(d.pop("function"))

        type_ = d.pop("type", UNSET)

        tool_function = cls(
            function=function,
            type_=type_,
        )

        tool_function.additional_properties = d
        return tool_function

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

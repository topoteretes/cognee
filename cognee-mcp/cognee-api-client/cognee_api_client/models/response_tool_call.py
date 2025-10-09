from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.function_call import FunctionCall
    from ..models.tool_call_output import ToolCallOutput


T = TypeVar("T", bound="ResponseToolCall")


@_attrs_define
class ResponseToolCall:
    """Tool call in a response

    Attributes:
        function (FunctionCall): Function call made by the assistant
        id (Union[Unset, str]):
        type_ (Union[Unset, str]):  Default: 'function'.
        output (Union['ToolCallOutput', None, Unset]):
    """

    function: "FunctionCall"
    id: Union[Unset, str] = UNSET
    type_: Union[Unset, str] = "function"
    output: Union["ToolCallOutput", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.tool_call_output import ToolCallOutput

        function = self.function.to_dict()

        id = self.id

        type_ = self.type_

        output: Union[None, Unset, dict[str, Any]]
        if isinstance(self.output, Unset):
            output = UNSET
        elif isinstance(self.output, ToolCallOutput):
            output = self.output.to_dict()
        else:
            output = self.output

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "function": function,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if type_ is not UNSET:
            field_dict["type"] = type_
        if output is not UNSET:
            field_dict["output"] = output

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.function_call import FunctionCall
        from ..models.tool_call_output import ToolCallOutput

        d = dict(src_dict)
        function = FunctionCall.from_dict(d.pop("function"))

        id = d.pop("id", UNSET)

        type_ = d.pop("type", UNSET)

        def _parse_output(data: object) -> Union["ToolCallOutput", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_type_0 = ToolCallOutput.from_dict(data)

                return output_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ToolCallOutput", None, Unset], data)

        output = _parse_output(d.pop("output", UNSET))

        response_tool_call = cls(
            function=function,
            id=id,
            type_=type_,
            output=output,
        )

        response_tool_call.additional_properties = d
        return response_tool_call

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

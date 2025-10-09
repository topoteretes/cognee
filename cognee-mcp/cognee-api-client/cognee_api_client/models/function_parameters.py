from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.function_parameters_properties import FunctionParametersProperties


T = TypeVar("T", bound="FunctionParameters")


@_attrs_define
class FunctionParameters:
    """JSON Schema for function parameters

    Attributes:
        properties (FunctionParametersProperties):
        type_ (Union[Unset, str]):  Default: 'object'.
        required (Union[None, Unset, list[str]]):
    """

    properties: "FunctionParametersProperties"
    type_: Union[Unset, str] = "object"
    required: Union[None, Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        properties = self.properties.to_dict()

        type_ = self.type_

        required: Union[None, Unset, list[str]]
        if isinstance(self.required, Unset):
            required = UNSET
        elif isinstance(self.required, list):
            required = self.required

        else:
            required = self.required

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "properties": properties,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if required is not UNSET:
            field_dict["required"] = required

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.function_parameters_properties import FunctionParametersProperties

        d = dict(src_dict)
        properties = FunctionParametersProperties.from_dict(d.pop("properties"))

        type_ = d.pop("type", UNSET)

        def _parse_required(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                required_type_0 = cast(list[str], data)

                return required_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        required = _parse_required(d.pop("required", UNSET))

        function_parameters = cls(
            properties=properties,
            type_=type_,
            required=required,
        )

        function_parameters.additional_properties = d
        return function_parameters

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

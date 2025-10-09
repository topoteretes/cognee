from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.function_parameters import FunctionParameters


T = TypeVar("T", bound="Function")


@_attrs_define
class Function:
    """Function definition compatible with OpenAI's format

    Attributes:
        name (str):
        description (str):
        parameters (FunctionParameters): JSON Schema for function parameters
    """

    name: str
    description: str
    parameters: "FunctionParameters"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        description = self.description

        parameters = self.parameters.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.function_parameters import FunctionParameters

        d = dict(src_dict)
        name = d.pop("name")

        description = d.pop("description")

        parameters = FunctionParameters.from_dict(d.pop("parameters"))

        function = cls(
            name=name,
            description=description,
            parameters=parameters,
        )

        function.additional_properties = d
        return function

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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.function_parameters_properties_additional_property import (
        FunctionParametersPropertiesAdditionalProperty,
    )


T = TypeVar("T", bound="FunctionParametersProperties")


@_attrs_define
class FunctionParametersProperties:
    """ """

    additional_properties: dict[str, "FunctionParametersPropertiesAdditionalProperty"] = _attrs_field(
        init=False, factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.function_parameters_properties_additional_property import (
            FunctionParametersPropertiesAdditionalProperty,
        )

        d = dict(src_dict)
        function_parameters_properties = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = FunctionParametersPropertiesAdditionalProperty.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        function_parameters_properties.additional_properties = additional_properties
        return function_parameters_properties

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "FunctionParametersPropertiesAdditionalProperty":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "FunctionParametersPropertiesAdditionalProperty") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.config_choice import ConfigChoice


T = TypeVar("T", bound="VectorDBConfigOutputDTO")


@_attrs_define
class VectorDBConfigOutputDTO:
    """
    Attributes:
        api_key (str):
        url (str):
        provider (str):
        providers (list['ConfigChoice']):
    """

    api_key: str
    url: str
    provider: str
    providers: list["ConfigChoice"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        api_key = self.api_key

        url = self.url

        provider = self.provider

        providers = []
        for providers_item_data in self.providers:
            providers_item = providers_item_data.to_dict()
            providers.append(providers_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "apiKey": api_key,
                "url": url,
                "provider": provider,
                "providers": providers,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_choice import ConfigChoice

        d = dict(src_dict)
        api_key = d.pop("apiKey")

        url = d.pop("url")

        provider = d.pop("provider")

        providers = []
        _providers = d.pop("providers")
        for providers_item_data in _providers:
            providers_item = ConfigChoice.from_dict(providers_item_data)

            providers.append(providers_item)

        vector_db_config_output_dto = cls(
            api_key=api_key,
            url=url,
            provider=provider,
            providers=providers,
        )

        vector_db_config_output_dto.additional_properties = d
        return vector_db_config_output_dto

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

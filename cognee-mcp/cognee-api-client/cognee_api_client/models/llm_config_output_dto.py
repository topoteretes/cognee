from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.config_choice import ConfigChoice
    from ..models.llm_config_output_dto_models import LLMConfigOutputDTOModels


T = TypeVar("T", bound="LLMConfigOutputDTO")


@_attrs_define
class LLMConfigOutputDTO:
    """
    Attributes:
        api_key (str):
        model (str):
        provider (str):
        endpoint (Union[None, str]):
        api_version (Union[None, str]):
        models (LLMConfigOutputDTOModels):
        providers (list['ConfigChoice']):
    """

    api_key: str
    model: str
    provider: str
    endpoint: Union[None, str]
    api_version: Union[None, str]
    models: "LLMConfigOutputDTOModels"
    providers: list["ConfigChoice"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        api_key = self.api_key

        model = self.model

        provider = self.provider

        endpoint: Union[None, str]
        endpoint = self.endpoint

        api_version: Union[None, str]
        api_version = self.api_version

        models = self.models.to_dict()

        providers = []
        for providers_item_data in self.providers:
            providers_item = providers_item_data.to_dict()
            providers.append(providers_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "apiKey": api_key,
                "model": model,
                "provider": provider,
                "endpoint": endpoint,
                "apiVersion": api_version,
                "models": models,
                "providers": providers,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_choice import ConfigChoice
        from ..models.llm_config_output_dto_models import LLMConfigOutputDTOModels

        d = dict(src_dict)
        api_key = d.pop("apiKey")

        model = d.pop("model")

        provider = d.pop("provider")

        def _parse_endpoint(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        endpoint = _parse_endpoint(d.pop("endpoint"))

        def _parse_api_version(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        api_version = _parse_api_version(d.pop("apiVersion"))

        models = LLMConfigOutputDTOModels.from_dict(d.pop("models"))

        providers = []
        _providers = d.pop("providers")
        for providers_item_data in _providers:
            providers_item = ConfigChoice.from_dict(providers_item_data)

            providers.append(providers_item)

        llm_config_output_dto = cls(
            api_key=api_key,
            model=model,
            provider=provider,
            endpoint=endpoint,
            api_version=api_version,
            models=models,
            providers=providers,
        )

        llm_config_output_dto.additional_properties = d
        return llm_config_output_dto

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

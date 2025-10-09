from collections.abc import Mapping
from typing import Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="LLMConfigInputDTO")


@_attrs_define
class LLMConfigInputDTO:
    """
    Attributes:
        provider (Union[Literal['anthropic'], Literal['gemini'], Literal['ollama'], Literal['openai']]):
        model (str):
        api_key (str):
    """

    provider: Union[Literal["anthropic"], Literal["gemini"], Literal["ollama"], Literal["openai"]]
    model: str
    api_key: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider: Union[Literal["anthropic"], Literal["gemini"], Literal["ollama"], Literal["openai"]]
        provider = self.provider

        model = self.model

        api_key = self.api_key

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "model": model,
                "apiKey": api_key,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_provider(
            data: object,
        ) -> Union[Literal["anthropic"], Literal["gemini"], Literal["ollama"], Literal["openai"]]:
            provider_type_0 = cast(Literal["openai"], data)
            if provider_type_0 != "openai":
                raise ValueError(f"provider_type_0 must match const 'openai', got '{provider_type_0}'")
            return provider_type_0
            provider_type_1 = cast(Literal["ollama"], data)
            if provider_type_1 != "ollama":
                raise ValueError(f"provider_type_1 must match const 'ollama', got '{provider_type_1}'")
            return provider_type_1
            provider_type_2 = cast(Literal["anthropic"], data)
            if provider_type_2 != "anthropic":
                raise ValueError(f"provider_type_2 must match const 'anthropic', got '{provider_type_2}'")
            return provider_type_2
            provider_type_3 = cast(Literal["gemini"], data)
            if provider_type_3 != "gemini":
                raise ValueError(f"provider_type_3 must match const 'gemini', got '{provider_type_3}'")
            return provider_type_3

        provider = _parse_provider(d.pop("provider"))

        model = d.pop("model")

        api_key = d.pop("apiKey")

        llm_config_input_dto = cls(
            provider=provider,
            model=model,
            api_key=api_key,
        )

        llm_config_input_dto.additional_properties = d
        return llm_config_input_dto

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

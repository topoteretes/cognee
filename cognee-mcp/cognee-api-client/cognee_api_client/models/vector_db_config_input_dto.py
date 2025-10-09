from collections.abc import Mapping
from typing import Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="VectorDBConfigInputDTO")


@_attrs_define
class VectorDBConfigInputDTO:
    """
    Attributes:
        provider (Union[Literal['chromadb'], Literal['lancedb'], Literal['pgvector']]):
        url (str):
        api_key (str):
    """

    provider: Union[Literal["chromadb"], Literal["lancedb"], Literal["pgvector"]]
    url: str
    api_key: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider: Union[Literal["chromadb"], Literal["lancedb"], Literal["pgvector"]]
        provider = self.provider

        url = self.url

        api_key = self.api_key

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "url": url,
                "apiKey": api_key,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_provider(data: object) -> Union[Literal["chromadb"], Literal["lancedb"], Literal["pgvector"]]:
            provider_type_0 = cast(Literal["lancedb"], data)
            if provider_type_0 != "lancedb":
                raise ValueError(f"provider_type_0 must match const 'lancedb', got '{provider_type_0}'")
            return provider_type_0
            provider_type_1 = cast(Literal["chromadb"], data)
            if provider_type_1 != "chromadb":
                raise ValueError(f"provider_type_1 must match const 'chromadb', got '{provider_type_1}'")
            return provider_type_1
            provider_type_2 = cast(Literal["pgvector"], data)
            if provider_type_2 != "pgvector":
                raise ValueError(f"provider_type_2 must match const 'pgvector', got '{provider_type_2}'")
            return provider_type_2

        provider = _parse_provider(d.pop("provider"))

        url = d.pop("url")

        api_key = d.pop("apiKey")

        vector_db_config_input_dto = cls(
            provider=provider,
            url=url,
            api_key=api_key,
        )

        vector_db_config_input_dto.additional_properties = d
        return vector_db_config_input_dto

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

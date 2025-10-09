from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.llm_config_output_dto import LLMConfigOutputDTO
    from ..models.vector_db_config_output_dto import VectorDBConfigOutputDTO


T = TypeVar("T", bound="SettingsDTO")


@_attrs_define
class SettingsDTO:
    """
    Attributes:
        llm (LLMConfigOutputDTO):
        vector_db (VectorDBConfigOutputDTO):
    """

    llm: "LLMConfigOutputDTO"
    vector_db: "VectorDBConfigOutputDTO"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        llm = self.llm.to_dict()

        vector_db = self.vector_db.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "llm": llm,
                "vectorDb": vector_db,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.llm_config_output_dto import LLMConfigOutputDTO
        from ..models.vector_db_config_output_dto import VectorDBConfigOutputDTO

        d = dict(src_dict)
        llm = LLMConfigOutputDTO.from_dict(d.pop("llm"))

        vector_db = VectorDBConfigOutputDTO.from_dict(d.pop("vectorDb"))

        settings_dto = cls(
            llm=llm,
            vector_db=vector_db,
        )

        settings_dto.additional_properties = d
        return settings_dto

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

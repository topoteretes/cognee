from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.llm_config_input_dto import LLMConfigInputDTO
    from ..models.vector_db_config_input_dto import VectorDBConfigInputDTO


T = TypeVar("T", bound="SettingsPayloadDTO")


@_attrs_define
class SettingsPayloadDTO:
    """
    Attributes:
        llm (Union['LLMConfigInputDTO', None, Unset]):
        vector_db (Union['VectorDBConfigInputDTO', None, Unset]):
    """

    llm: Union["LLMConfigInputDTO", None, Unset] = UNSET
    vector_db: Union["VectorDBConfigInputDTO", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.llm_config_input_dto import LLMConfigInputDTO
        from ..models.vector_db_config_input_dto import VectorDBConfigInputDTO

        llm: Union[None, Unset, dict[str, Any]]
        if isinstance(self.llm, Unset):
            llm = UNSET
        elif isinstance(self.llm, LLMConfigInputDTO):
            llm = self.llm.to_dict()
        else:
            llm = self.llm

        vector_db: Union[None, Unset, dict[str, Any]]
        if isinstance(self.vector_db, Unset):
            vector_db = UNSET
        elif isinstance(self.vector_db, VectorDBConfigInputDTO):
            vector_db = self.vector_db.to_dict()
        else:
            vector_db = self.vector_db

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if llm is not UNSET:
            field_dict["llm"] = llm
        if vector_db is not UNSET:
            field_dict["vectorDb"] = vector_db

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.llm_config_input_dto import LLMConfigInputDTO
        from ..models.vector_db_config_input_dto import VectorDBConfigInputDTO

        d = dict(src_dict)

        def _parse_llm(data: object) -> Union["LLMConfigInputDTO", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                llm_type_0 = LLMConfigInputDTO.from_dict(data)

                return llm_type_0
            except:  # noqa: E722
                pass
            return cast(Union["LLMConfigInputDTO", None, Unset], data)

        llm = _parse_llm(d.pop("llm", UNSET))

        def _parse_vector_db(data: object) -> Union["VectorDBConfigInputDTO", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                vector_db_type_0 = VectorDBConfigInputDTO.from_dict(data)

                return vector_db_type_0
            except:  # noqa: E722
                pass
            return cast(Union["VectorDBConfigInputDTO", None, Unset], data)

        vector_db = _parse_vector_db(d.pop("vectorDb", UNSET))

        settings_payload_dto = cls(
            llm=llm,
            vector_db=vector_db,
        )

        settings_payload_dto.additional_properties = d
        return settings_payload_dto

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

from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="CodePipelineIndexPayloadDTO")


@_attrs_define
class CodePipelineIndexPayloadDTO:
    """
    Attributes:
        repo_path (str):
        include_docs (Union[Unset, bool]):  Default: False.
    """

    repo_path: str
    include_docs: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        repo_path = self.repo_path

        include_docs = self.include_docs

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "repoPath": repo_path,
            }
        )
        if include_docs is not UNSET:
            field_dict["includeDocs"] = include_docs

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        repo_path = d.pop("repoPath")

        include_docs = d.pop("includeDocs", UNSET)

        code_pipeline_index_payload_dto = cls(
            repo_path=repo_path,
            include_docs=include_docs,
        )

        code_pipeline_index_payload_dto.additional_properties = d
        return code_pipeline_index_payload_dto

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

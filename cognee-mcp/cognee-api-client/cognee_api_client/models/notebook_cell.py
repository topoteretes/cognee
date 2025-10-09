from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.notebook_cell_type import NotebookCellType

T = TypeVar("T", bound="NotebookCell")


@_attrs_define
class NotebookCell:
    """
    Attributes:
        id (UUID):
        type_ (NotebookCellType):
        name (str):
        content (str):
    """

    id: UUID
    type_: NotebookCellType
    name: str
    content: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        type_ = self.type_.value

        name = self.name

        content = self.content

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "type": type_,
                "name": name,
                "content": content,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = UUID(d.pop("id"))

        type_ = NotebookCellType(d.pop("type"))

        name = d.pop("name")

        content = d.pop("content")

        notebook_cell = cls(
            id=id,
            type_=type_,
            name=name,
            content=content,
        )

        notebook_cell.additional_properties = d
        return notebook_cell

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

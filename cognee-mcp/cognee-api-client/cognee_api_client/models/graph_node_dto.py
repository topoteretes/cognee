from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.graph_node_dto_properties import GraphNodeDTOProperties


T = TypeVar("T", bound="GraphNodeDTO")


@_attrs_define
class GraphNodeDTO:
    """
    Attributes:
        id (UUID):
        label (str):
        properties (GraphNodeDTOProperties):
    """

    id: UUID
    label: str
    properties: "GraphNodeDTOProperties"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        label = self.label

        properties = self.properties.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "label": label,
                "properties": properties,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.graph_node_dto_properties import GraphNodeDTOProperties

        d = dict(src_dict)
        id = UUID(d.pop("id"))

        label = d.pop("label")

        properties = GraphNodeDTOProperties.from_dict(d.pop("properties"))

        graph_node_dto = cls(
            id=id,
            label=label,
            properties=properties,
        )

        graph_node_dto.additional_properties = d
        return graph_node_dto

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

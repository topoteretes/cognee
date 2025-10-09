from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.graph_edge_dto import GraphEdgeDTO
    from ..models.graph_node_dto import GraphNodeDTO


T = TypeVar("T", bound="GraphDTO")


@_attrs_define
class GraphDTO:
    """
    Attributes:
        nodes (list['GraphNodeDTO']):
        edges (list['GraphEdgeDTO']):
    """

    nodes: list["GraphNodeDTO"]
    edges: list["GraphEdgeDTO"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        nodes = []
        for nodes_item_data in self.nodes:
            nodes_item = nodes_item_data.to_dict()
            nodes.append(nodes_item)

        edges = []
        for edges_item_data in self.edges:
            edges_item = edges_item_data.to_dict()
            edges.append(edges_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "nodes": nodes,
                "edges": edges,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.graph_edge_dto import GraphEdgeDTO
        from ..models.graph_node_dto import GraphNodeDTO

        d = dict(src_dict)
        nodes = []
        _nodes = d.pop("nodes")
        for nodes_item_data in _nodes:
            nodes_item = GraphNodeDTO.from_dict(nodes_item_data)

            nodes.append(nodes_item)

        edges = []
        _edges = d.pop("edges")
        for edges_item_data in _edges:
            edges_item = GraphEdgeDTO.from_dict(edges_item_data)

            edges.append(edges_item)

        graph_dto = cls(
            nodes=nodes,
            edges=edges,
        )

        graph_dto.additional_properties = d
        return graph_dto

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

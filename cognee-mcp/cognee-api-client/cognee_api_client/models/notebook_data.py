from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.notebook_cell import NotebookCell


T = TypeVar("T", bound="NotebookData")


@_attrs_define
class NotebookData:
    """
    Attributes:
        name (Union[None, str]):
        cells (Union[None, Unset, list['NotebookCell']]):
    """

    name: Union[None, str]
    cells: Union[None, Unset, list["NotebookCell"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name: Union[None, str]
        name = self.name

        cells: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.cells, Unset):
            cells = UNSET
        elif isinstance(self.cells, list):
            cells = []
            for cells_type_0_item_data in self.cells:
                cells_type_0_item = cells_type_0_item_data.to_dict()
                cells.append(cells_type_0_item)

        else:
            cells = self.cells

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if cells is not UNSET:
            field_dict["cells"] = cells

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.notebook_cell import NotebookCell

        d = dict(src_dict)

        def _parse_name(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        name = _parse_name(d.pop("name"))

        def _parse_cells(data: object) -> Union[None, Unset, list["NotebookCell"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                cells_type_0 = []
                _cells_type_0 = data
                for cells_type_0_item_data in _cells_type_0:
                    cells_type_0_item = NotebookCell.from_dict(cells_type_0_item_data)

                    cells_type_0.append(cells_type_0_item)

                return cells_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["NotebookCell"]], data)

        cells = _parse_cells(d.pop("cells", UNSET))

        notebook_data = cls(
            name=name,
            cells=cells,
        )

        notebook_data.additional_properties = d
        return notebook_data

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

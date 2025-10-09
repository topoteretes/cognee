from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.combined_search_result_context import CombinedSearchResultContext
    from ..models.combined_search_result_graphs_type_0 import CombinedSearchResultGraphsType0
    from ..models.search_result_dataset import SearchResultDataset


T = TypeVar("T", bound="CombinedSearchResult")


@_attrs_define
class CombinedSearchResult:
    """
    Attributes:
        result (Union[Any, None]):
        context (CombinedSearchResultContext):
        graphs (Union['CombinedSearchResultGraphsType0', None, Unset]):
        datasets (Union[None, Unset, list['SearchResultDataset']]):
    """

    result: Union[Any, None]
    context: "CombinedSearchResultContext"
    graphs: Union["CombinedSearchResultGraphsType0", None, Unset] = UNSET
    datasets: Union[None, Unset, list["SearchResultDataset"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.combined_search_result_graphs_type_0 import CombinedSearchResultGraphsType0

        result: Union[Any, None]
        result = self.result

        context = self.context.to_dict()

        graphs: Union[None, Unset, dict[str, Any]]
        if isinstance(self.graphs, Unset):
            graphs = UNSET
        elif isinstance(self.graphs, CombinedSearchResultGraphsType0):
            graphs = self.graphs.to_dict()
        else:
            graphs = self.graphs

        datasets: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.datasets, Unset):
            datasets = UNSET
        elif isinstance(self.datasets, list):
            datasets = []
            for datasets_type_0_item_data in self.datasets:
                datasets_type_0_item = datasets_type_0_item_data.to_dict()
                datasets.append(datasets_type_0_item)

        else:
            datasets = self.datasets

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "result": result,
                "context": context,
            }
        )
        if graphs is not UNSET:
            field_dict["graphs"] = graphs
        if datasets is not UNSET:
            field_dict["datasets"] = datasets

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.combined_search_result_context import CombinedSearchResultContext
        from ..models.combined_search_result_graphs_type_0 import CombinedSearchResultGraphsType0
        from ..models.search_result_dataset import SearchResultDataset

        d = dict(src_dict)

        def _parse_result(data: object) -> Union[Any, None]:
            if data is None:
                return data
            return cast(Union[Any, None], data)

        result = _parse_result(d.pop("result"))

        context = CombinedSearchResultContext.from_dict(d.pop("context"))

        def _parse_graphs(data: object) -> Union["CombinedSearchResultGraphsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                graphs_type_0 = CombinedSearchResultGraphsType0.from_dict(data)

                return graphs_type_0
            except:  # noqa: E722
                pass
            return cast(Union["CombinedSearchResultGraphsType0", None, Unset], data)

        graphs = _parse_graphs(d.pop("graphs", UNSET))

        def _parse_datasets(data: object) -> Union[None, Unset, list["SearchResultDataset"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                datasets_type_0 = []
                _datasets_type_0 = data
                for datasets_type_0_item_data in _datasets_type_0:
                    datasets_type_0_item = SearchResultDataset.from_dict(datasets_type_0_item_data)

                    datasets_type_0.append(datasets_type_0_item)

                return datasets_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["SearchResultDataset"]], data)

        datasets = _parse_datasets(d.pop("datasets", UNSET))

        combined_search_result = cls(
            result=result,
            context=context,
            graphs=graphs,
            datasets=datasets,
        )

        combined_search_result.additional_properties = d
        return combined_search_result

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

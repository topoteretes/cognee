from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.search_type import SearchType
from ..types import UNSET, Unset

T = TypeVar("T", bound="SearchPayloadDTO")


@_attrs_define
class SearchPayloadDTO:
    """
    Attributes:
        search_type (Union[Unset, SearchType]):
        datasets (Union[None, Unset, list[str]]):
        dataset_ids (Union[None, Unset, list[UUID]]):
        query (Union[Unset, str]):  Default: 'What is in the document?'.
        system_prompt (Union[None, Unset, str]):  Default: 'Answer the question using the provided context. Be as brief
            as possible.'.
        node_name (Union[None, Unset, list[str]]):
        top_k (Union[None, Unset, int]):  Default: 10.
        only_context (Union[Unset, bool]):  Default: False.
        use_combined_context (Union[Unset, bool]):  Default: False.
    """

    search_type: Union[Unset, SearchType] = UNSET
    datasets: Union[None, Unset, list[str]] = UNSET
    dataset_ids: Union[None, Unset, list[UUID]] = UNSET
    query: Union[Unset, str] = "What is in the document?"
    system_prompt: Union[None, Unset, str] = "Answer the question using the provided context. Be as brief as possible."
    node_name: Union[None, Unset, list[str]] = UNSET
    top_k: Union[None, Unset, int] = 10
    only_context: Union[Unset, bool] = False
    use_combined_context: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        search_type: Union[Unset, str] = UNSET
        if not isinstance(self.search_type, Unset):
            search_type = self.search_type.value

        datasets: Union[None, Unset, list[str]]
        if isinstance(self.datasets, Unset):
            datasets = UNSET
        elif isinstance(self.datasets, list):
            datasets = self.datasets

        else:
            datasets = self.datasets

        dataset_ids: Union[None, Unset, list[str]]
        if isinstance(self.dataset_ids, Unset):
            dataset_ids = UNSET
        elif isinstance(self.dataset_ids, list):
            dataset_ids = []
            for dataset_ids_type_0_item_data in self.dataset_ids:
                dataset_ids_type_0_item = str(dataset_ids_type_0_item_data)
                dataset_ids.append(dataset_ids_type_0_item)

        else:
            dataset_ids = self.dataset_ids

        query = self.query

        system_prompt: Union[None, Unset, str]
        if isinstance(self.system_prompt, Unset):
            system_prompt = UNSET
        else:
            system_prompt = self.system_prompt

        node_name: Union[None, Unset, list[str]]
        if isinstance(self.node_name, Unset):
            node_name = UNSET
        elif isinstance(self.node_name, list):
            node_name = self.node_name

        else:
            node_name = self.node_name

        top_k: Union[None, Unset, int]
        if isinstance(self.top_k, Unset):
            top_k = UNSET
        else:
            top_k = self.top_k

        only_context = self.only_context

        use_combined_context = self.use_combined_context

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if search_type is not UNSET:
            field_dict["searchType"] = search_type
        if datasets is not UNSET:
            field_dict["datasets"] = datasets
        if dataset_ids is not UNSET:
            field_dict["datasetIds"] = dataset_ids
        if query is not UNSET:
            field_dict["query"] = query
        if system_prompt is not UNSET:
            field_dict["systemPrompt"] = system_prompt
        if node_name is not UNSET:
            field_dict["nodeName"] = node_name
        if top_k is not UNSET:
            field_dict["topK"] = top_k
        if only_context is not UNSET:
            field_dict["onlyContext"] = only_context
        if use_combined_context is not UNSET:
            field_dict["useCombinedContext"] = use_combined_context

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        _search_type = d.pop("searchType", UNSET)
        search_type: Union[Unset, SearchType]
        if isinstance(_search_type, Unset):
            search_type = UNSET
        else:
            search_type = SearchType(_search_type)

        def _parse_datasets(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                datasets_type_0 = cast(list[str], data)

                return datasets_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        datasets = _parse_datasets(d.pop("datasets", UNSET))

        def _parse_dataset_ids(data: object) -> Union[None, Unset, list[UUID]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                dataset_ids_type_0 = []
                _dataset_ids_type_0 = data
                for dataset_ids_type_0_item_data in _dataset_ids_type_0:
                    dataset_ids_type_0_item = UUID(dataset_ids_type_0_item_data)

                    dataset_ids_type_0.append(dataset_ids_type_0_item)

                return dataset_ids_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[UUID]], data)

        dataset_ids = _parse_dataset_ids(d.pop("datasetIds", UNSET))

        query = d.pop("query", UNSET)

        def _parse_system_prompt(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        system_prompt = _parse_system_prompt(d.pop("systemPrompt", UNSET))

        def _parse_node_name(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                node_name_type_0 = cast(list[str], data)

                return node_name_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        node_name = _parse_node_name(d.pop("nodeName", UNSET))

        def _parse_top_k(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        top_k = _parse_top_k(d.pop("topK", UNSET))

        only_context = d.pop("onlyContext", UNSET)

        use_combined_context = d.pop("useCombinedContext", UNSET)

        search_payload_dto = cls(
            search_type=search_type,
            datasets=datasets,
            dataset_ids=dataset_ids,
            query=query,
            system_prompt=system_prompt,
            node_name=node_name,
            top_k=top_k,
            only_context=only_context,
            use_combined_context=use_combined_context,
        )

        search_payload_dto.additional_properties = d
        return search_payload_dto

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

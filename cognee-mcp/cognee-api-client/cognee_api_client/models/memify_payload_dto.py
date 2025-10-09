from collections.abc import Mapping
from typing import Any, Literal, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="MemifyPayloadDTO")


@_attrs_define
class MemifyPayloadDTO:
    """
    Attributes:
        extraction_tasks (Union[None, Unset, list[str]]):
        enrichment_tasks (Union[None, Unset, list[str]]):
        data (Union[None, Unset, str]):  Default: ''.
        dataset_name (Union[None, Unset, str]):
        dataset_id (Union[Literal[''], None, UUID, Unset]):
        node_name (Union[None, Unset, list[str]]):
        run_in_background (Union[None, Unset, bool]):  Default: False.
    """

    extraction_tasks: Union[None, Unset, list[str]] = UNSET
    enrichment_tasks: Union[None, Unset, list[str]] = UNSET
    data: Union[None, Unset, str] = ""
    dataset_name: Union[None, Unset, str] = UNSET
    dataset_id: Union[Literal[""], None, UUID, Unset] = UNSET
    node_name: Union[None, Unset, list[str]] = UNSET
    run_in_background: Union[None, Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        extraction_tasks: Union[None, Unset, list[str]]
        if isinstance(self.extraction_tasks, Unset):
            extraction_tasks = UNSET
        elif isinstance(self.extraction_tasks, list):
            extraction_tasks = self.extraction_tasks

        else:
            extraction_tasks = self.extraction_tasks

        enrichment_tasks: Union[None, Unset, list[str]]
        if isinstance(self.enrichment_tasks, Unset):
            enrichment_tasks = UNSET
        elif isinstance(self.enrichment_tasks, list):
            enrichment_tasks = self.enrichment_tasks

        else:
            enrichment_tasks = self.enrichment_tasks

        data: Union[None, Unset, str]
        if isinstance(self.data, Unset):
            data = UNSET
        else:
            data = self.data

        dataset_name: Union[None, Unset, str]
        if isinstance(self.dataset_name, Unset):
            dataset_name = UNSET
        else:
            dataset_name = self.dataset_name

        dataset_id: Union[Literal[""], None, Unset, str]
        if isinstance(self.dataset_id, Unset):
            dataset_id = UNSET
        elif isinstance(self.dataset_id, UUID):
            dataset_id = str(self.dataset_id)
        else:
            dataset_id = self.dataset_id

        node_name: Union[None, Unset, list[str]]
        if isinstance(self.node_name, Unset):
            node_name = UNSET
        elif isinstance(self.node_name, list):
            node_name = self.node_name

        else:
            node_name = self.node_name

        run_in_background: Union[None, Unset, bool]
        if isinstance(self.run_in_background, Unset):
            run_in_background = UNSET
        else:
            run_in_background = self.run_in_background

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if extraction_tasks is not UNSET:
            field_dict["extractionTasks"] = extraction_tasks
        if enrichment_tasks is not UNSET:
            field_dict["enrichmentTasks"] = enrichment_tasks
        if data is not UNSET:
            field_dict["data"] = data
        if dataset_name is not UNSET:
            field_dict["datasetName"] = dataset_name
        if dataset_id is not UNSET:
            field_dict["datasetId"] = dataset_id
        if node_name is not UNSET:
            field_dict["nodeName"] = node_name
        if run_in_background is not UNSET:
            field_dict["runInBackground"] = run_in_background

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_extraction_tasks(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                extraction_tasks_type_0 = cast(list[str], data)

                return extraction_tasks_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        extraction_tasks = _parse_extraction_tasks(d.pop("extractionTasks", UNSET))

        def _parse_enrichment_tasks(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                enrichment_tasks_type_0 = cast(list[str], data)

                return enrichment_tasks_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        enrichment_tasks = _parse_enrichment_tasks(d.pop("enrichmentTasks", UNSET))

        def _parse_data(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        data = _parse_data(d.pop("data", UNSET))

        def _parse_dataset_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        dataset_name = _parse_dataset_name(d.pop("datasetName", UNSET))

        def _parse_dataset_id(data: object) -> Union[Literal[""], None, UUID, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                dataset_id_type_0 = UUID(data)

                return dataset_id_type_0
            except:  # noqa: E722
                pass
            dataset_id_type_1 = cast(Literal[""], data)
            if dataset_id_type_1 != "":
                raise ValueError(f"datasetId_type_1 must match const '', got '{dataset_id_type_1}'")
            return dataset_id_type_1
            return cast(Union[Literal[""], None, UUID, Unset], data)

        dataset_id = _parse_dataset_id(d.pop("datasetId", UNSET))

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

        def _parse_run_in_background(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        run_in_background = _parse_run_in_background(d.pop("runInBackground", UNSET))

        memify_payload_dto = cls(
            extraction_tasks=extraction_tasks,
            enrichment_tasks=enrichment_tasks,
            data=data,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            node_name=node_name,
            run_in_background=run_in_background,
        )

        memify_payload_dto.additional_properties = d
        return memify_payload_dto

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

from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="CognifyPayloadDTO")


@_attrs_define
class CognifyPayloadDTO:
    """
    Attributes:
        datasets (Union[None, Unset, list[str]]):
        dataset_ids (Union[None, Unset, list[UUID]]):
        run_in_background (Union[None, Unset, bool]):  Default: False.
        custom_prompt (Union[None, Unset, str]): Custom prompt for entity extraction and graph generation Default: ''.
    """

    datasets: Union[None, Unset, list[str]] = UNSET
    dataset_ids: Union[None, Unset, list[UUID]] = UNSET
    run_in_background: Union[None, Unset, bool] = False
    custom_prompt: Union[None, Unset, str] = ""
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
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

        run_in_background: Union[None, Unset, bool]
        if isinstance(self.run_in_background, Unset):
            run_in_background = UNSET
        else:
            run_in_background = self.run_in_background

        custom_prompt: Union[None, Unset, str]
        if isinstance(self.custom_prompt, Unset):
            custom_prompt = UNSET
        else:
            custom_prompt = self.custom_prompt

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if datasets is not UNSET:
            field_dict["datasets"] = datasets
        if dataset_ids is not UNSET:
            field_dict["datasetIds"] = dataset_ids
        if run_in_background is not UNSET:
            field_dict["runInBackground"] = run_in_background
        if custom_prompt is not UNSET:
            field_dict["customPrompt"] = custom_prompt

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

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

        def _parse_run_in_background(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        run_in_background = _parse_run_in_background(d.pop("runInBackground", UNSET))

        def _parse_custom_prompt(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        custom_prompt = _parse_custom_prompt(d.pop("customPrompt", UNSET))

        cognify_payload_dto = cls(
            datasets=datasets,
            dataset_ids=dataset_ids,
            run_in_background=run_in_background,
            custom_prompt=custom_prompt,
        )

        cognify_payload_dto.additional_properties = d
        return cognify_payload_dto

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

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.pipeline_run_status import PipelineRunStatus

T = TypeVar("T", bound="GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet")


@_attrs_define
class GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet:
    """ """

    additional_properties: dict[str, PipelineRunStatus] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.value

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        get_dataset_status_api_v1_datasets_status_get_response_get_dataset_status_api_v1_datasets_status_get = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = PipelineRunStatus(prop_dict)

            additional_properties[prop_name] = additional_property

        get_dataset_status_api_v1_datasets_status_get_response_get_dataset_status_api_v1_datasets_status_get.additional_properties = additional_properties
        return get_dataset_status_api_v1_datasets_status_get_response_get_dataset_status_api_v1_datasets_status_get

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> PipelineRunStatus:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: PipelineRunStatus) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

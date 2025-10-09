import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="DataDTO")


@_attrs_define
class DataDTO:
    """
    Attributes:
        id (UUID):
        name (str):
        created_at (datetime.datetime):
        extension (str):
        mime_type (str):
        raw_data_location (str):
        dataset_id (UUID):
        updated_at (Union[None, Unset, datetime.datetime]):
    """

    id: UUID
    name: str
    created_at: datetime.datetime
    extension: str
    mime_type: str
    raw_data_location: str
    dataset_id: UUID
    updated_at: Union[None, Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        name = self.name

        created_at = self.created_at.isoformat()

        extension = self.extension

        mime_type = self.mime_type

        raw_data_location = self.raw_data_location

        dataset_id = str(self.dataset_id)

        updated_at: Union[None, Unset, str]
        if isinstance(self.updated_at, Unset):
            updated_at = UNSET
        elif isinstance(self.updated_at, datetime.datetime):
            updated_at = self.updated_at.isoformat()
        else:
            updated_at = self.updated_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "createdAt": created_at,
                "extension": extension,
                "mimeType": mime_type,
                "rawDataLocation": raw_data_location,
                "datasetId": dataset_id,
            }
        )
        if updated_at is not UNSET:
            field_dict["updatedAt"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = UUID(d.pop("id"))

        name = d.pop("name")

        created_at = isoparse(d.pop("createdAt"))

        extension = d.pop("extension")

        mime_type = d.pop("mimeType")

        raw_data_location = d.pop("rawDataLocation")

        dataset_id = UUID(d.pop("datasetId"))

        def _parse_updated_at(data: object) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                updated_at_type_0 = isoparse(data)

                return updated_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        updated_at = _parse_updated_at(d.pop("updatedAt", UNSET))

        data_dto = cls(
            id=id,
            name=name,
            created_at=created_at,
            extension=extension,
            mime_type=mime_type,
            raw_data_location=raw_data_location,
            dataset_id=dataset_id,
            updated_at=updated_at,
        )

        data_dto.additional_properties = d
        return data_dto

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

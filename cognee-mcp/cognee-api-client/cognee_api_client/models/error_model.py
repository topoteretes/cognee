from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.error_model_detail_type_1 import ErrorModelDetailType1


T = TypeVar("T", bound="ErrorModel")


@_attrs_define
class ErrorModel:
    """
    Attributes:
        detail (Union['ErrorModelDetailType1', str]):
    """

    detail: Union["ErrorModelDetailType1", str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.error_model_detail_type_1 import ErrorModelDetailType1

        detail: Union[dict[str, Any], str]
        if isinstance(self.detail, ErrorModelDetailType1):
            detail = self.detail.to_dict()
        else:
            detail = self.detail

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "detail": detail,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.error_model_detail_type_1 import ErrorModelDetailType1

        d = dict(src_dict)

        def _parse_detail(data: object) -> Union["ErrorModelDetailType1", str]:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                detail_type_1 = ErrorModelDetailType1.from_dict(data)

                return detail_type_1
            except:  # noqa: E722
                pass
            return cast(Union["ErrorModelDetailType1", str], data)

        detail = _parse_detail(d.pop("detail"))

        error_model = cls(
            detail=detail,
        )

        error_model.additional_properties = d
        return error_model

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

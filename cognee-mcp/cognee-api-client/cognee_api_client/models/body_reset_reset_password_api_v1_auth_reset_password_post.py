from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="BodyResetResetPasswordApiV1AuthResetPasswordPost")


@_attrs_define
class BodyResetResetPasswordApiV1AuthResetPasswordPost:
    """
    Attributes:
        token (str):
        password (str):
    """

    token: str
    password: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        token = self.token

        password = self.password

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "token": token,
                "password": password,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        token = d.pop("token")

        password = d.pop("password")

        body_reset_reset_password_api_v1_auth_reset_password_post = cls(
            token=token,
            password=password,
        )

        body_reset_reset_password_api_v1_auth_reset_password_post.additional_properties = d
        return body_reset_reset_password_api_v1_auth_reset_password_post

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

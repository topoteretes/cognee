from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.chat_usage import ChatUsage
    from ..models.response_body_metadata import ResponseBodyMetadata
    from ..models.response_tool_call import ResponseToolCall


T = TypeVar("T", bound="ResponseBody")


@_attrs_define
class ResponseBody:
    """Response body for the new responses endpoint

    Attributes:
        model (str):
        tool_calls (list['ResponseToolCall']):
        id (Union[Unset, str]):
        created (Union[Unset, int]):
        object_ (Union[Unset, str]):  Default: 'response'.
        status (Union[Unset, str]):  Default: 'completed'.
        usage (Union['ChatUsage', None, Unset]):
        metadata (Union[Unset, ResponseBodyMetadata]):
    """

    model: str
    tool_calls: list["ResponseToolCall"]
    id: Union[Unset, str] = UNSET
    created: Union[Unset, int] = UNSET
    object_: Union[Unset, str] = "response"
    status: Union[Unset, str] = "completed"
    usage: Union["ChatUsage", None, Unset] = UNSET
    metadata: Union[Unset, "ResponseBodyMetadata"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.chat_usage import ChatUsage

        model = self.model

        tool_calls = []
        for tool_calls_item_data in self.tool_calls:
            tool_calls_item = tool_calls_item_data.to_dict()
            tool_calls.append(tool_calls_item)

        id = self.id

        created = self.created

        object_ = self.object_

        status = self.status

        usage: Union[None, Unset, dict[str, Any]]
        if isinstance(self.usage, Unset):
            usage = UNSET
        elif isinstance(self.usage, ChatUsage):
            usage = self.usage.to_dict()
        else:
            usage = self.usage

        metadata: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "model": model,
                "toolCalls": tool_calls,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if created is not UNSET:
            field_dict["created"] = created
        if object_ is not UNSET:
            field_dict["object"] = object_
        if status is not UNSET:
            field_dict["status"] = status
        if usage is not UNSET:
            field_dict["usage"] = usage
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.chat_usage import ChatUsage
        from ..models.response_body_metadata import ResponseBodyMetadata
        from ..models.response_tool_call import ResponseToolCall

        d = dict(src_dict)
        model = d.pop("model")

        tool_calls = []
        _tool_calls = d.pop("toolCalls")
        for tool_calls_item_data in _tool_calls:
            tool_calls_item = ResponseToolCall.from_dict(tool_calls_item_data)

            tool_calls.append(tool_calls_item)

        id = d.pop("id", UNSET)

        created = d.pop("created", UNSET)

        object_ = d.pop("object", UNSET)

        status = d.pop("status", UNSET)

        def _parse_usage(data: object) -> Union["ChatUsage", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                usage_type_0 = ChatUsage.from_dict(data)

                return usage_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ChatUsage", None, Unset], data)

        usage = _parse_usage(d.pop("usage", UNSET))

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, ResponseBodyMetadata]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = ResponseBodyMetadata.from_dict(_metadata)

        response_body = cls(
            model=model,
            tool_calls=tool_calls,
            id=id,
            created=created,
            object_=object_,
            status=status,
            usage=usage,
            metadata=metadata,
        )

        response_body.additional_properties = d
        return response_body

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

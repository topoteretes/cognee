from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.cognee_model import CogneeModel
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.response_request_tool_choice_type_1 import ResponseRequestToolChoiceType1
    from ..models.tool_function import ToolFunction


T = TypeVar("T", bound="ResponseRequest")


@_attrs_define
class ResponseRequest:
    """Request body for the new responses endpoint (OpenAI Responses API format)

    Attributes:
        input_ (str):
        model (Union[Unset, CogneeModel]): Enum for supported model types
        tools (Union[None, Unset, list['ToolFunction']]):
        tool_choice (Union['ResponseRequestToolChoiceType1', None, Unset, str]):  Default: 'auto'.
        user (Union[None, Unset, str]):
        temperature (Union[None, Unset, float]):  Default: 1.0.
        max_completion_tokens (Union[None, Unset, int]):
    """

    input_: str
    model: Union[Unset, CogneeModel] = UNSET
    tools: Union[None, Unset, list["ToolFunction"]] = UNSET
    tool_choice: Union["ResponseRequestToolChoiceType1", None, Unset, str] = "auto"
    user: Union[None, Unset, str] = UNSET
    temperature: Union[None, Unset, float] = 1.0
    max_completion_tokens: Union[None, Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.response_request_tool_choice_type_1 import ResponseRequestToolChoiceType1

        input_ = self.input_

        model: Union[Unset, str] = UNSET
        if not isinstance(self.model, Unset):
            model = self.model.value

        tools: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.tools, Unset):
            tools = UNSET
        elif isinstance(self.tools, list):
            tools = []
            for tools_type_0_item_data in self.tools:
                tools_type_0_item = tools_type_0_item_data.to_dict()
                tools.append(tools_type_0_item)

        else:
            tools = self.tools

        tool_choice: Union[None, Unset, dict[str, Any], str]
        if isinstance(self.tool_choice, Unset):
            tool_choice = UNSET
        elif isinstance(self.tool_choice, ResponseRequestToolChoiceType1):
            tool_choice = self.tool_choice.to_dict()
        else:
            tool_choice = self.tool_choice

        user: Union[None, Unset, str]
        if isinstance(self.user, Unset):
            user = UNSET
        else:
            user = self.user

        temperature: Union[None, Unset, float]
        if isinstance(self.temperature, Unset):
            temperature = UNSET
        else:
            temperature = self.temperature

        max_completion_tokens: Union[None, Unset, int]
        if isinstance(self.max_completion_tokens, Unset):
            max_completion_tokens = UNSET
        else:
            max_completion_tokens = self.max_completion_tokens

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input": input_,
            }
        )
        if model is not UNSET:
            field_dict["model"] = model
        if tools is not UNSET:
            field_dict["tools"] = tools
        if tool_choice is not UNSET:
            field_dict["toolChoice"] = tool_choice
        if user is not UNSET:
            field_dict["user"] = user
        if temperature is not UNSET:
            field_dict["temperature"] = temperature
        if max_completion_tokens is not UNSET:
            field_dict["maxCompletionTokens"] = max_completion_tokens

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.response_request_tool_choice_type_1 import ResponseRequestToolChoiceType1
        from ..models.tool_function import ToolFunction

        d = dict(src_dict)
        input_ = d.pop("input")

        _model = d.pop("model", UNSET)
        model: Union[Unset, CogneeModel]
        if isinstance(_model, Unset):
            model = UNSET
        else:
            model = CogneeModel(_model)

        def _parse_tools(data: object) -> Union[None, Unset, list["ToolFunction"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tools_type_0 = []
                _tools_type_0 = data
                for tools_type_0_item_data in _tools_type_0:
                    tools_type_0_item = ToolFunction.from_dict(tools_type_0_item_data)

                    tools_type_0.append(tools_type_0_item)

                return tools_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["ToolFunction"]], data)

        tools = _parse_tools(d.pop("tools", UNSET))

        def _parse_tool_choice(data: object) -> Union["ResponseRequestToolChoiceType1", None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                tool_choice_type_1 = ResponseRequestToolChoiceType1.from_dict(data)

                return tool_choice_type_1
            except:  # noqa: E722
                pass
            return cast(Union["ResponseRequestToolChoiceType1", None, Unset, str], data)

        tool_choice = _parse_tool_choice(d.pop("toolChoice", UNSET))

        def _parse_user(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        user = _parse_user(d.pop("user", UNSET))

        def _parse_temperature(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        temperature = _parse_temperature(d.pop("temperature", UNSET))

        def _parse_max_completion_tokens(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        max_completion_tokens = _parse_max_completion_tokens(d.pop("maxCompletionTokens", UNSET))

        response_request = cls(
            input_=input_,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            user=user,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )

        response_request.additional_properties = d
        return response_request

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

import time
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from enum import Enum

from cognee.api.DTO import InDTO, OutDTO


class CogneeModel(str, Enum):
    """Enum for supported model types"""

    COGNEEV1 = "cognee-v1"


class FunctionParameters(BaseModel):
    """JSON Schema for function parameters"""

    type: str = "object"
    properties: Dict[str, Dict[str, Any]]
    required: Optional[List[str]] = None


class Function(BaseModel):
    """Function definition compatible with OpenAI's format"""

    name: str
    description: str
    parameters: FunctionParameters


class ToolFunction(BaseModel):
    """Tool function wrapper (for OpenAI compatibility)"""

    type: str = "function"
    function: Function


class FunctionCall(BaseModel):
    """Function call made by the assistant"""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call made by the assistant"""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex}")
    type: str = "function"
    function: FunctionCall


class ChatUsage(BaseModel):
    """Token usage information"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseRequest(InDTO):
    """Request body for the new responses endpoint (OpenAI Responses API format)"""

    model: CogneeModel = CogneeModel.COGNEEV1
    input: str
    tools: Optional[List[ToolFunction]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    user: Optional[str] = None
    temperature: Optional[float] = 1.0
    max_completion_tokens: Optional[int] = None


class ToolCallOutput(BaseModel):
    """Output of a tool call in the responses API"""

    status: str = "success"  # success/error
    data: Optional[Dict[str, Any]] = None


class ResponseToolCall(BaseModel):
    """Tool call in a response"""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex}")
    type: str = "function"
    function: FunctionCall
    output: Optional[ToolCallOutput] = None


class ResponseBody(OutDTO):
    """Response body for the new responses endpoint"""

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    object: str = "response"
    status: str = "completed"
    tool_calls: List[ResponseToolCall]
    usage: Optional[ChatUsage] = None
    metadata: Dict[str, Any] = None

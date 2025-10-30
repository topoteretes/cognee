from typing import Literal
from pydantic import BaseModel


class MCPServerUrlConfig(BaseModel):
    url: str
    transport: Literal["http", "sse"]

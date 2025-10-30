from typing import List
from pydantic import BaseModel


class MCPServerCommandConfig(BaseModel):
    command: str
    args: List[str]

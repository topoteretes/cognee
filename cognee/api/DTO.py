from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel, to_snake
from typing import Optional


class OutDTO(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class InDTO(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class ErrorResponse(OutDTO):
    error: str
    detail: Optional[str] = None

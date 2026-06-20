"""Shared API DTO field types aligned with core function signatures."""

from typing import Annotated, Any, Optional, Union
from uuid import UUID

from pydantic import BeforeValidator


def _coerce_optional_str_list(value: Any) -> Optional[list[str]]:
    """Accept a single string or list of strings; normalize to list or None."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped == "" else [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"Expected str, list[str], or null; got {type(value).__name__}")


def _coerce_optional_uuid_list(value: Any) -> Optional[list[UUID]]:
    """Accept a single UUID or list of UUIDs; normalize to list or None."""
    if value is None:
        return None
    if isinstance(value, UUID):
        return [value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return [UUID(stripped)]
    if isinstance(value, (list, tuple)):
        return [item if isinstance(item, UUID) else UUID(str(item)) for item in value]
    raise ValueError(f"Expected UUID, list[UUID], or null; got {type(value).__name__}")


OptionalStringList = Annotated[
    Optional[list[str]],
    BeforeValidator(_coerce_optional_str_list),
]

OptionalUUIDList = Annotated[
    Optional[list[UUID]],
    BeforeValidator(_coerce_optional_uuid_list),
]

# Type aliases for annotations that also document accepted wire formats.
DatasetNamesInput = Union[str, list[str], None]
DatasetIdsInput = Union[UUID, list[UUID], None]
NodeNamesInput = Union[str, list[str], None]

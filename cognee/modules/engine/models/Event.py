from typing import Any, Optional

from pydantic import SkipValidation

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.Interval import Interval
from cognee.modules.engine.models.Timestamp import Timestamp


class Event(DataPoint):
    name: str
    description: Optional[str] = None
    at: Optional[Timestamp] = None
    during: Optional[Interval] = None
    location: Optional[str] = None
    attributes: SkipValidation[Any] = None

    metadata: dict = {"index_fields": ["name"]}

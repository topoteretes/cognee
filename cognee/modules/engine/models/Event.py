from typing import Any

from pydantic import SkipValidation

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.Interval import Interval
from cognee.modules.engine.models.Timestamp import Timestamp


class Event(DataPoint):
    name: str
    description: str | None = None
    at: Timestamp | None = None
    during: Interval | None = None
    location: str | None = None
    attributes: SkipValidation[Any] = None

    metadata: dict = {"index_fields": ["name"]}

from pydantic import Field
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.Timestamp import Timestamp


class Interval(DataPoint):
    time_from: Timestamp = Field(...)
    time_to: Timestamp = Field(...)

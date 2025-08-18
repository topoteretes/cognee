from cognee.exceptions import (
    CogneeValidationError,
)
from fastapi import status


class InvalidDataPointsInAddDataPointsError(CogneeValidationError):
    def __init__(self, detail: str):
        super().__init__(
            message=f"Invalid data_points: {detail}",
            name="InvalidDataPointsInAddDataPointsError",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

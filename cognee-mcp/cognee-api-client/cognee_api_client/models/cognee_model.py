from enum import Enum


class CogneeModel(str, Enum):
    COGNEE_V1 = "cognee-v1"

    def __str__(self) -> str:
        return str(self.value)

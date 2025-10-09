from enum import Enum


class NotebookCellType(str, Enum):
    CODE = "code"
    MARKDOWN = "markdown"

    def __str__(self) -> str:
        return str(self.value)

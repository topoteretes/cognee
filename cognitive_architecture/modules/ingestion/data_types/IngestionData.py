from typing import Protocol, BinaryIO

class IngestionData(Protocol):
    data: str | BinaryIO = None
    metadata: dict = None

    def get_data(self):
        pass

    def get_extension(self):
        pass

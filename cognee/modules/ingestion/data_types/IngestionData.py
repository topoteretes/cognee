from typing import Protocol, BinaryIO, Union


class IngestionData(Protocol):
    data: Union[str, BinaryIO] = None

    def get_data(self):
        raise NotImplementedError("Subclasses must implement get_data()")

    def get_identifier(self):
        raise NotImplementedError("Subclasses must implement get_identifier()")

    def get_metadata(self):
        raise NotImplementedError("Subclasses must implement get_metadata()")

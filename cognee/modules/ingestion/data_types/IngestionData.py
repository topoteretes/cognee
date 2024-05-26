from typing import Protocol, BinaryIO, Union


class IngestionData(Protocol):
    data: Union[str, BinaryIO] = None

    def get_data(self):
        raise NotImplementedError()

    def get_identifier(self):
        raise NotImplementedError()

    def get_metadata(self):
        raise NotImplementedError()

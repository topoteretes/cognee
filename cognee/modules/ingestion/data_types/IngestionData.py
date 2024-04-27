from typing import Protocol, BinaryIO

from typing import Union, Protocol, BinaryIO

class IngestionData(Protocol):
    data: Union[str, BinaryIO] = None

    def get_data(self) -> None:
        raise NotImplementedError()

    def get_identifier(self) -> None:
        raise NotImplementedError()

    def get_metadata(self) -> None:
        raise NotImplementedError()


from io import BufferedReader


class FileBufferedReader(BufferedReader):
    def __init__(self, file_obj, name: str) -> None:
        super().__init__(file_obj)
        self._file = file_obj
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def read(self, size: int | None = -1):
        data = self._file.read(size)
        return data

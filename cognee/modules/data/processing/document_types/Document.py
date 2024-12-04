from cognee.infrastructure.engine import DataPoint

class Document(DataPoint):
    type: str
    name: str
    raw_data_location: str

    def read(self, chunk_size: int) -> str:
        pass

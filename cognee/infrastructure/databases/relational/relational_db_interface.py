from abc import abstractmethod
from typing import Protocol, TypeVar, Type, List

RowDataType = TypeVar('RowDataType')

class RelationalDBInterface(Protocol):
    @abstractmethod
    async def create_database(self, database_name: str, database_path: str): raise NotImplementedError

    @abstractmethod
    async def create_table(self, table_name: str, table_config: object): raise NotImplementedError

    @abstractmethod
    async def add_row(self, table_name: str, row_data: Type[RowDataType]): raise NotImplementedError

    @abstractmethod
    async def add_rows(self, table_name: str, rows_data: List[Type[RowDataType]]): raise NotImplementedError

    @abstractmethod
    async def get_row(self, table_name: str, row_id: str): raise NotImplementedError

    @abstractmethod
    async def update_row(self, table_name: str, row_id: str, row_data: Type[RowDataType]): raise NotImplementedError

    @abstractmethod
    async def delete_row(self, table_name: str, row_id: str): raise NotImplementedError

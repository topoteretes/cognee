from os import path
from contextlib import asynccontextmanager

from cognee.infrastructure.files.storage import get_file_storage


@asynccontextmanager
async def open_data_file(file_path: str, mode: str = "rb", encoding: str = None, **kwargs):
    file_dir_path = path.dirname(file_path)
    file_name = path.basename(file_path)

    file_storage = get_file_storage(file_dir_path)

    async with file_storage.open(file_name, mode=mode, encoding=encoding, **kwargs) as file:
        yield file

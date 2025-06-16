from os import path
from typing import Optional
from contextlib import contextmanager
from cognee.api.v1.add.config import get_s3_config
from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage


@contextmanager
def open_data_file(file_path: str, mode: str = "rb", encoding: Optional[str] = None, **kwargs):
    if file_path.startswith("s3://"):
        s3_config = get_s3_config()
        if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
            import s3fs

            fs = s3fs.S3FileSystem(
                key=s3_config.aws_access_key_id, secret=s3_config.aws_secret_access_key, anon=False
            )
        else:
            raise ValueError("S3 credentials are not set in the configuration.")

        file_dir_path = path.dirname(file_path)
        file_name = path.basename(file_path)
        file_storage = S3FileStorage(file_dir_path)

        if "b" in mode:
            with file_storage.open(file_name, mode=mode, **kwargs) as file:
                if not hasattr(file, "name") or not file.name:
                    file.name = file_name
                    yield file

        else:
            with file_storage.open(file_name, mode=mode, **kwargs) as file:
                yield file

    else:
        file_dir_path = path.dirname(file_path)
        file_name = path.basename(file_path)
        file_storage = LocalFileStorage(file_dir_path)

        return file_storage.open(file_name, mode=mode, encoding=encoding, **kwargs)
